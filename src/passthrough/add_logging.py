#!/usr/bin/env python3
"""Transform hip_intercept.c to add logging to all pass-through functions."""
import re
import sys

def transform(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    result = []
    i = 0
    
    # Track the insertion point for logging infrastructure
    insert_after_line = None
    for idx, line in enumerate(lines):
        if 'static void ensure_init(void)' in line:
            insert_after_line = idx
            break
    
    if insert_after_line is None:
        print("ERROR: Could not find ensure_init()")
        sys.exit(1)
    
    # Find the closing brace of ensure_init
    brace_depth = 0
    for idx in range(insert_after_line, len(lines)):
        if '{' in lines[idx]:
            brace_depth += 1
        if '}' in lines[idx]:
            brace_depth -= 1
            if brace_depth == 0:
                insert_after_line = idx
                break
    
    logging_infrastructure = """
//===----------------------------------------------------------------------===//
// Built-in Pass-through Logging
//
// Logs all HIP API calls that bypass the function table (i.e., are not
// handled by the interceptor library). This ensures complete coverage.
//===----------------------------------------------------------------------===//

#include <stdarg.h>
#include <time.h>

static FILE* g_pt_log_file = NULL;
static int g_pt_log_level = 0;
static pthread_mutex_t g_pt_log_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_pt_log_initialized = 0;

static void pt_log_init(void) {
  if (g_pt_log_initialized) return;
  g_pt_log_initialized = 1;
  
  const char* log_path = getenv("HIP_LOG_FILE");
  if (log_path && *log_path) {
    // Open in append mode so we don't clobber interceptor logs
    g_pt_log_file = fopen(log_path, "a");
    if (!g_pt_log_file) g_pt_log_file = stderr;
  } else {
    g_pt_log_file = stderr;
  }
  
  const char* level_str = getenv("HIP_LOG_LEVEL");
  if (level_str) g_pt_log_level = atoi(level_str);
}

static void pt_log(int level, const char* fmt, ...) {
  if (!g_pt_log_initialized) pt_log_init();
  if (level > g_pt_log_level || !g_pt_log_file) return;
  
  pthread_mutex_lock(&g_pt_log_mutex);
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  fprintf(g_pt_log_file, "[%ld.%06ld] ", (long)ts.tv_sec, ts.tv_nsec / 1000);
  va_list args;
  va_start(args, fmt);
  vfprintf(g_pt_log_file, fmt, args);
  va_end(args);
  fprintf(g_pt_log_file, "\\n");
  fflush(g_pt_log_file);
  pthread_mutex_unlock(&g_pt_log_mutex);
}

"""

    # Build the output with logging infrastructure inserted
    for idx, line in enumerate(lines):
        result.append(line)
        if idx == insert_after_line:
            result.append(logging_infrastructure)
    
    output = ''.join(result)
    result_lines = output.split('\n')
    final_lines = []
    
    i = 0
    while i < len(result_lines):
        line = result_lines[i]
        
        # Skip FWD macro lines
        if line.startswith('FWD'):
            final_lines.append(line)
            i += 1
            continue
        
        # Look for the dlsym pattern to get the function name
        dlsym_match = re.search(r'dlsym\(g_backend_lib,\s*"(\w+)"\)', line)
        if dlsym_match:
            func_name = dlsym_match.group(1)
            final_lines.append(line)
            i += 1
            
            # Look ahead for the return statement (within next 3 lines)
            found_return = False
            for j in range(min(3, len(result_lines) - i)):
                peek_line = result_lines[i + j]
                
                # Pattern: "  return fn ? fn(...) : 1;" (with args)
                return_match = re.match(r'^(\s*)return fn \? fn\((.+)\) : (\S+);$', peek_line)
                if return_match:
                    indent = return_match.group(1)
                    args = return_match.group(2)
                    default_val = return_match.group(3)
                    
                    for k in range(j):
                        final_lines.append(result_lines[i + k])
                    
                    # Check if return type is const char* (default is "unknown" or similar string)
                    if default_val.startswith('"'):
                        final_lines.append(f'{indent}const char* _ret = fn ? fn({args}) : {default_val};')
                        final_lines.append(f'{indent}pt_log(2, "{func_name}() -> %s", _ret ? _ret : "(null)");')
                    else:
                        final_lines.append(f'{indent}hipError_t _ret = fn ? fn({args}) : {default_val};')
                        final_lines.append(f'{indent}pt_log(2, "{func_name}() -> %d", _ret);')
                    final_lines.append(f'{indent}return _ret;')
                    
                    i += j + 1
                    found_return = True
                    break
                
                # Pattern: "  return fn ? fn() : 1;" (no args)
                return_noarg_match = re.match(r'^(\s*)return fn \? fn\(\) : (\S+);$', peek_line)
                if return_noarg_match:
                    indent = return_noarg_match.group(1)
                    default_val = return_noarg_match.group(2)
                    
                    for k in range(j):
                        final_lines.append(result_lines[i + k])
                    
                    if default_val.startswith('"'):
                        final_lines.append(f'{indent}const char* _ret = fn ? fn() : {default_val};')
                        final_lines.append(f'{indent}pt_log(2, "{func_name}() -> %s", _ret ? _ret : "(null)");')
                    else:
                        final_lines.append(f'{indent}hipError_t _ret = fn ? fn() : {default_val};')
                        final_lines.append(f'{indent}pt_log(2, "{func_name}() -> %d", _ret);')
                    final_lines.append(f'{indent}return _ret;')
                    
                    i += j + 1
                    found_return = True
                    break
                
                # Pattern for void functions: "  if (fn) fn(...);"
                void_match = re.match(r'^(\s*)if \(fn\) fn\((.+)\);$', peek_line)
                if void_match:
                    indent = void_match.group(1)
                    args = void_match.group(2)
                    
                    for k in range(j):
                        final_lines.append(result_lines[i + k])
                    
                    final_lines.append(peek_line)
                    final_lines.append(f'{indent}pt_log(2, "{func_name}()");')
                    
                    i += j + 1
                    found_return = True
                    break
            
            if not found_return:
                pass
            continue
        
        final_lines.append(line)
        i += 1
    
    # Now enhance the important functions with detailed parameter logging
    output_text = '\n'.join(final_lines)
    
    # Enhance memory functions with size/pointer info
    enhancements = {
        # Memory allocation
        'hipMallocAsync': ('dev_ptr, size, stream',
            'pt_log(2, "hipMallocAsync(size=%zu, stream=%p) -> ptr=%p, ret=%d", size, (void*)stream, dev_ptr ? *dev_ptr : NULL, _ret);'),
        'hipFreeAsync': ('dev_ptr, stream',
            'pt_log(2, "hipFreeAsync(ptr=%p, stream=%p) -> %d", dev_ptr, (void*)stream, _ret);'),
        'hipMallocManaged': ('dev_ptr, size, flags',
            'pt_log(2, "hipMallocManaged(size=%zu, flags=0x%x) -> ptr=%p, ret=%d", size, flags, dev_ptr ? *dev_ptr : NULL, _ret);'),
        'hipMallocPitch': ('ptr, pitch, width, height',
            'pt_log(2, "hipMallocPitch(width=%zu, height=%zu) -> ptr=%p, pitch=%zu, ret=%d", width, height, ptr ? *ptr : NULL, pitch ? *pitch : 0, _ret);'),
        'hipExtMallocWithFlags': ('ptr, sizeBytes, flags',
            'pt_log(2, "hipExtMallocWithFlags(size=%zu, flags=0x%x) -> ptr=%p, ret=%d", sizeBytes, flags, ptr ? *ptr : NULL, _ret);'),
        'hipMemAlloc': ('dptr, size',
            'pt_log(2, "hipMemAlloc(size=%zu) -> ptr=%p, ret=%d", size, dptr ? *dptr : NULL, _ret);'),
        'hipMemFree': ('dptr',
            'pt_log(2, "hipMemFree(ptr=%p) -> %d", dptr, _ret);'),
        
        # Memory copy variants
        'hipMemcpyWithStream': ('dst, src, sizeBytes, kind, stream',
            'pt_log(2, "hipMemcpyWithStream(dst=%p, src=%p, size=%zu, kind=%d, stream=%p) -> %d", dst, src, sizeBytes, kind, (void*)stream, _ret);'),
        'hipMemcpy2D': ('dst, dpitch, src, spitch, width, height, kind',
            'pt_log(2, "hipMemcpy2D(dst=%p, src=%p, width=%zu, height=%zu, kind=%d) -> %d", dst, src, width, height, kind, _ret);'),
        'hipMemcpy2DAsync': ('dst, dpitch, src, spitch, width, height, kind, stream',
            'pt_log(2, "hipMemcpy2DAsync(dst=%p, src=%p, width=%zu, height=%zu, kind=%d, stream=%p) -> %d", dst, src, width, height, kind, (void*)stream, _ret);'),
        'hipMemcpyDtoD': ('dst, src, sizeBytes',
            'pt_log(2, "hipMemcpyDtoD(dst=%p, src=%p, size=%zu) -> %d", dst, src, sizeBytes, _ret);'),
        'hipMemcpyDtoDAsync': ('dst, src, sizeBytes, stream',
            'pt_log(2, "hipMemcpyDtoDAsync(dst=%p, src=%p, size=%zu, stream=%p) -> %d", dst, src, sizeBytes, (void*)stream, _ret);'),
        'hipMemcpyDtoH': ('dst, src, sizeBytes',
            'pt_log(2, "hipMemcpyDtoH(dst=%p, src=%p, size=%zu) -> %d", dst, src, sizeBytes, _ret);'),
        'hipMemcpyDtoHAsync': ('dst, src, sizeBytes, stream',
            'pt_log(2, "hipMemcpyDtoHAsync(dst=%p, src=%p, size=%zu, stream=%p) -> %d", dst, src, sizeBytes, (void*)stream, _ret);'),
        'hipMemcpyHtoD': ('dst, src, sizeBytes',
            'pt_log(2, "hipMemcpyHtoD(dst=%p, src=%p, size=%zu) -> %d", dst, src, sizeBytes, _ret);'),
        'hipMemcpyHtoDAsync': ('dst, src, sizeBytes, stream',
            'pt_log(2, "hipMemcpyHtoDAsync(dst=%p, src=%p, size=%zu, stream=%p) -> %d", dst, src, sizeBytes, (void*)stream, _ret);'),
        'hipMemcpyPeer': ('dst, dstDeviceId, src, srcDeviceId, sizeBytes',
            'pt_log(2, "hipMemcpyPeer(dst=%p, dstDev=%d, src=%p, srcDev=%d, size=%zu) -> %d", dst, dstDeviceId, src, srcDeviceId, sizeBytes, _ret);'),
        
        # Memset variants
        'hipMemset2D': ('dst, pitch, value, width, height',
            'pt_log(2, "hipMemset2D(dst=%p, value=0x%02x, width=%zu, height=%zu) -> %d", dst, value, width, height, _ret);'),
        'hipMemset2DAsync': ('dst, pitch, value, width, height, stream',
            'pt_log(2, "hipMemset2DAsync(dst=%p, value=0x%02x, width=%zu, height=%zu, stream=%p) -> %d", dst, value, width, height, (void*)stream, _ret);'),
        'hipMemsetD8': ('dst, value, count',
            'pt_log(2, "hipMemsetD8(dst=%p, value=0x%02x, count=%zu) -> %d", dst, value, count, _ret);'),
        'hipMemsetD8Async': ('dst, value, count, stream',
            'pt_log(2, "hipMemsetD8Async(dst=%p, value=0x%02x, count=%zu, stream=%p) -> %d", dst, value, count, (void*)stream, _ret);'),
        'hipMemsetD16': ('dst, value, count',
            'pt_log(2, "hipMemsetD16(dst=%p, value=0x%04x, count=%zu) -> %d", dst, value, count, _ret);'),
        'hipMemsetD16Async': ('dst, value, count, stream',
            'pt_log(2, "hipMemsetD16Async(dst=%p, value=0x%04x, count=%zu, stream=%p) -> %d", dst, value, count, (void*)stream, _ret);'),
        'hipMemsetD32': ('dst, value, count',
            'pt_log(2, "hipMemsetD32(dst=%p, value=0x%08x, count=%zu) -> %d", dst, value, count, _ret);'),
        'hipMemsetD32Async': ('dst, value, count, stream',
            'pt_log(2, "hipMemsetD32Async(dst=%p, value=0x%08x, count=%zu, stream=%p) -> %d", dst, value, count, (void*)stream, _ret);'),
        
        # Module
        'hipModuleLoadDataEx': ('module, image, numOptions, options, optionValues',
            'pt_log(2, "hipModuleLoadDataEx(image=%p, numOpts=%u) -> module=%p, ret=%d", image, numOptions, module ? *module : NULL, _ret);'),
        
        # Context
        'hipCtxCreate': ('ctx, flags, device',
            'pt_log(2, "hipCtxCreate(flags=0x%x, device=%p) -> ctx=%p, ret=%d", flags, device, ctx ? *ctx : NULL, _ret);'),
        'hipCtxSetCurrent': ('ctx',
            'pt_log(2, "hipCtxSetCurrent(ctx=%p) -> %d", ctx, _ret);'),
        'hipCtxGetCurrent': ('ctx',
            'pt_log(2, "hipCtxGetCurrent() -> ctx=%p, ret=%d", ctx ? *ctx : NULL, _ret);'),
        'hipDeviceComputeCapability': ('major, minor, device',
            'pt_log(2, "hipDeviceComputeCapability(device=%p) -> major=%d, minor=%d, ret=%d", device, major ? *major : -1, minor ? *minor : -1, _ret);'),
        'hipDeviceTotalMem': ('bytes, device',
            'pt_log(2, "hipDeviceTotalMem(device=%p) -> bytes=%zu, ret=%d", device, bytes ? *bytes : 0, _ret);'),
        'hipDeviceGetPCIBusId': ('pciBusId, len, device',
            'pt_log(2, "hipDeviceGetPCIBusId(device=%d) -> id=%s, ret=%d", device, pciBusId ? pciBusId : "(null)", _ret);'),
        
        # Pointer queries
        'hipPointerGetAttributes': ('attributes, ptr',
            'pt_log(2, "hipPointerGetAttributes(ptr=%p) -> %d", ptr, _ret);'),
        
        # Stream capture
        'hipStreamBeginCapture': ('stream, mode',
            'pt_log(2, "hipStreamBeginCapture(stream=%p, mode=%d) -> %d", (void*)stream, mode, _ret);'),
        'hipStreamEndCapture': ('stream, pGraph',
            'pt_log(2, "hipStreamEndCapture(stream=%p) -> graph=%p, ret=%d", (void*)stream, pGraph ? *pGraph : NULL, _ret);'),
        
        # Graph
        'hipGraphCreate': ('pGraph, flags',
            'pt_log(2, "hipGraphCreate(flags=0x%x) -> graph=%p, ret=%d", flags, pGraph ? *pGraph : NULL, _ret);'),
        'hipGraphDestroy': ('graph',
            'pt_log(2, "hipGraphDestroy(graph=%p) -> %d", graph, _ret);'),
        'hipGraphLaunch': ('graphExec, stream',
            'pt_log(2, "hipGraphLaunch(exec=%p, stream=%p) -> %d", graphExec, (void*)stream, _ret);'),
        
        # Host registration
        'hipHostRegister': ('hostPtr, sizeBytes, flags',
            'pt_log(2, "hipHostRegister(ptr=%p, size=%zu, flags=0x%x) -> %d", hostPtr, sizeBytes, flags, _ret);'),
        'hipHostUnregister': ('hostPtr',
            'pt_log(2, "hipHostUnregister(ptr=%p) -> %d", hostPtr, _ret);'),
        'hipHostGetDevicePointer': ('devPtr, hostPtr, flags',
            'pt_log(2, "hipHostGetDevicePointer(hostPtr=%p) -> devPtr=%p, ret=%d", hostPtr, devPtr ? *devPtr : NULL, _ret);'),
        
        # Occupancy
        'hipOccupancyMaxActiveBlocksPerMultiprocessor': ('numBlocks, f, blockSize, dynamicSMemSize',
            'pt_log(2, "hipOccupancyMaxActiveBlocksPerMultiprocessor(func=%p, blockSize=%d, dynMem=%zu) -> blocks=%d, ret=%d", f, blockSize, dynamicSMemSize, numBlocks ? *numBlocks : -1, _ret);'),
        'hipOccupancyMaxPotentialBlockSize': ('gridSize, blockSize, f, dynamicSMemSize, blockSizeLimit',
            'pt_log(2, "hipOccupancyMaxPotentialBlockSize(func=%p) -> grid=%d, block=%d, ret=%d", f, gridSize ? *gridSize : -1, blockSize ? *blockSize : -1, _ret);'),
        'hipModuleOccupancyMaxPotentialBlockSize': ('gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit',
            'pt_log(2, "hipModuleOccupancyMaxPotentialBlockSize(func=%p) -> grid=%d, block=%d, ret=%d", (void*)f, gridSize ? *gridSize : -1, blockSize ? *blockSize : -1, _ret);'),
        'hipModuleOccupancyMaxActiveBlocksPerMultiprocessor': ('numBlocks, f, blockSize, dynSharedMemPerBlk',
            'pt_log(2, "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(func=%p, blockSize=%d) -> blocks=%d, ret=%d", (void*)f, blockSize, numBlocks ? *numBlocks : -1, _ret);'),
        
        # Kernel launch
        'hipExtLaunchKernel': ('function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags',
            'pt_log(2, "hipExtLaunchKernel(func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), shared=%zu, stream=%p, flags=0x%x) -> %d", function_address, numBlocks.x, numBlocks.y, numBlocks.z, dimBlocks.x, dimBlocks.y, dimBlocks.z, sharedMemBytes, (void*)stream, flags, _ret);'),
        'hipLaunchCooperativeKernel': ('f, gridDim, blockDim, kernelParams, sharedMemBytes, stream',
            'pt_log(2, "hipLaunchCooperativeKernel(func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), shared=%u, stream=%p) -> %d", f, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMemBytes, (void*)stream, _ret);'),
        
        # Device limits
        'hipDeviceGetLimit': ('pValue, limit',
            'pt_log(2, "hipDeviceGetLimit(limit=%d) -> value=%zu, ret=%d", limit, pValue ? *pValue : 0, _ret);'),
        'hipDeviceSetLimit': ('limit, value',
            'pt_log(2, "hipDeviceSetLimit(limit=%d, value=%zu) -> %d", limit, value, _ret);'),
        'hipSetDeviceFlags': ('flags',
            'pt_log(2, "hipSetDeviceFlags(flags=0x%x) -> %d", flags, _ret);'),
        'hipGetDeviceFlags': ('flags',
            'pt_log(2, "hipGetDeviceFlags() -> flags=0x%x, ret=%d", flags ? *flags : 0, _ret);'),
        
        # IPC
        'hipIpcGetMemHandle': ('handle, devPtr',
            'pt_log(2, "hipIpcGetMemHandle(ptr=%p) -> %d", devPtr, _ret);'),
        'hipIpcOpenMemHandle': ('devPtr, handle, flags',
            'pt_log(2, "hipIpcOpenMemHandle(flags=0x%x) -> ptr=%p, ret=%d", flags, devPtr ? *devPtr : NULL, _ret);'),
        'hipIpcCloseMemHandle': ('devPtr',
            'pt_log(2, "hipIpcCloseMemHandle(ptr=%p) -> %d", devPtr, _ret);'),
        
        # Memory range
        'hipMemPrefetchAsync': ('dev_ptr, count, device, stream',
            'pt_log(2, "hipMemPrefetchAsync(ptr=%p, count=%zu, device=%d, stream=%p) -> %d", dev_ptr, count, device, (void*)stream, _ret);'),
        'hipMemAdvise': ('dev_ptr, count, advice, device',
            'pt_log(2, "hipMemAdvise(ptr=%p, count=%zu, advice=%d, device=%d) -> %d", dev_ptr, count, advice, device, _ret);'),
    }
    
    # Apply detailed enhancements
    for func_name, (_, detailed_log) in enhancements.items():
        # Replace the generic log with the detailed one
        generic_pattern = f'pt_log(2, "{func_name}() -> %d", _ret);'
        if generic_pattern in output_text:
            output_text = output_text.replace(generic_pattern, detailed_log, 1)
    
    with open(output_path, 'w') as f:
        f.write(output_text)
    
    # Count transformations
    count = output_text.count('pt_log(')
    print(f"Transformed {input_path} -> {output_path}")
    print(f"Total pt_log() calls: {count}")

if __name__ == '__main__':
    input_path = '/home/awoloszyn/Development/iree-stream/passthrough/hip_intercept.c'
    output_path = '/home/awoloszyn/Development/iree-stream/passthrough/hip_intercept.c.new'
    transform(input_path, output_path)
