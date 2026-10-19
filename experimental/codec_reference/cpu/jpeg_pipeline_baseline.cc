// Bounded public-libjpeg baseline; no production decoder substitution.
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/task/executor.h"
#include "iree/task/process.h"
#include "iree/task/topology.h"
#include "jpeglib.h"

namespace {

[[noreturn]] void Fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::exit(1);
}

void Check(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    Fail("task baseline failed");
  }
}

void JpegFatal(j_common_ptr codec) {
  codec->err->output_message(codec);
  Fail("JPEG oracle error");
}

void JpegMessage(j_common_ptr codec, int level) {
  if (level < 0) {
    codec->err->output_message(codec);
    Fail("JPEG oracle warning disqualifies this input");
  }
}

void InitializeError(jpeg_error_mgr* error) {
  jpeg_std_error(error);
  error->error_exit = JpegFatal;
  error->emit_message = JpegMessage;
}

uint64_t Hash(const uint8_t* data, size_t length) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (size_t i = 0; i < length; ++i) {
    hash = (hash ^ data[i]) * UINT64_C(1099511628211);
  }
  return hash;
}

struct Image {
  // Immutable compressed source loaded before any timed operation.
  std::vector<uint8_t> bytes;
  // JSON-safe basename chosen by the input manifest.
  std::string name;
  // Encoded horizontal sample count.
  uint32_t width = 0;
  // Encoded vertical sample count.
  uint32_t height = 0;
  // Exact host RGB8 destination extent.
  size_t pixel_bytes = 0;
  // Sequential or progressive marker information from the public header.
  bool progressive = false;
  // Restart interval in scan MCUs at the first scan.
  uint32_t restart_interval = 0;
  // Component sampling pairs in SOF order.
  std::vector<int> sampling;
};

Image ReadImage(const char* path) {
  Image image;
  image.name = std::filesystem::path(path).filename().string();
  if (image.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ"
                                   "RSTUVWXYZ0123456789-_.") !=
      std::string::npos) {
    Fail("input basename must be JSON-safe");
  }
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() <= 0) {
    Fail("cannot open input");
  }
  image.bytes.resize(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(image.bytes.data()), image.bytes.size());
  if (!file) {
    Fail("short input read");
  }
  jpeg_decompress_struct codec{};
  jpeg_error_mgr error;
  InitializeError(&error);
  codec.err = &error;
  jpeg_create_decompress(&codec);
  jpeg_mem_src(&codec, image.bytes.data(), image.bytes.size());
  if (jpeg_read_header(&codec, TRUE) != JPEG_HEADER_OK) {
    Fail("missing header");
  }
  image.width = codec.image_width;
  image.height = codec.image_height;
  image.pixel_bytes = size_t(image.width) * image.height * 3;
  if (!image.width || !image.height || image.pixel_bytes > 512 * 1024 * 1024) {
    Fail("fixture exceeds bounded output capacity");
  }
  image.progressive = codec.progressive_mode;
  image.restart_interval = codec.restart_interval;
  for (int i = 0; i < codec.num_components; ++i) {
    image.sampling.push_back(codec.comp_info[i].h_samp_factor);
    image.sampling.push_back(codec.comp_info[i].v_samp_factor);
  }
  jpeg_destroy_decompress(&codec);
  return image;
}

// Same task pool and scanline loop, with explicit public IDCT selection. Output
// allocation is caller-owned; libjpeg's create/start/finish/destroy stay timed.
void Decode(const Image& image, uint8_t* output) {
  jpeg_decompress_struct codec{};
  jpeg_error_mgr error;
  InitializeError(&error);
  codec.err = &error;
  jpeg_create_decompress(&codec);
  jpeg_mem_src(&codec, image.bytes.data(), image.bytes.size());
  jpeg_read_header(&codec, TRUE);
  codec.out_color_space = JCS_RGB;
  codec.dct_method = CODEC_REFERENCE_IDCT;
  jpeg_start_decompress(&codec);
  if (codec.output_width != image.width ||
      codec.output_height != image.height || codec.output_components != 3 ||
      codec.dct_method != CODEC_REFERENCE_IDCT || !codec.do_fancy_upsampling) {
    Fail("reference decode contract changed");
  }
  while (codec.output_scanline < codec.output_height) {
    JSAMPROW row = output + size_t(codec.output_scanline) * image.width * 3;
    if (jpeg_read_scanlines(&codec, &row, 1) != 1) {
      Fail("short scanline");
    }
  }
  if (!jpeg_finish_decompress(&codec)) {
    Fail("incomplete decode");
  }
  jpeg_destroy_decompress(&codec);
}

std::vector<uint16_t> Coefficients(const Image& image) {
  jpeg_decompress_struct codec{};
  jpeg_error_mgr error;
  InitializeError(&error);
  codec.err = &error;
  jpeg_create_decompress(&codec);
  jpeg_mem_src(&codec, image.bytes.data(), image.bytes.size());
  jpeg_read_header(&codec, TRUE);
  auto arrays = jpeg_read_coefficients(&codec);
  std::vector<uint16_t> result;
  for (int c = 0; c < codec.num_components; ++c) {
    const auto& component = codec.comp_info[c];
    if (component.width_in_blocks > 65535 ||
        component.height_in_blocks > 65535) {
      Fail("coefficient fixture extent exceeds signature encoding");
    }
    result.push_back(component.width_in_blocks);
    result.push_back(component.height_in_blocks);
    for (size_t k = 0; k < 64; ++k) {
      result.push_back(component.quant_table->quantval[k]);
    }
    for (JDIMENSION row = 0; row < component.height_in_blocks; ++row) {
      auto blocks = codec.mem->access_virt_barray(
          reinterpret_cast<j_common_ptr>(&codec), arrays[c], row, 1, FALSE);
      for (JDIMENSION column = 0; column < component.width_in_blocks;
           ++column) {
        for (size_t k = 0; k < 64; ++k) {
          result.push_back(blocks[0][column][k]);
        }
      }
    }
  }
  jpeg_finish_decompress(&codec);
  jpeg_destroy_decompress(&codec);
  return result;
}

void WriteSynthetic(const std::string& path, uint32_t width, uint32_t height) {
  jpeg_compress_struct codec{};
  jpeg_error_mgr error;
  InitializeError(&error);
  codec.err = &error;
  jpeg_create_compress(&codec);
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) {
    Fail("cannot create synthetic source");
  }
  jpeg_stdio_dest(&codec, file);
  codec.image_width = width;
  codec.image_height = height;
  codec.input_components = 3;
  codec.in_color_space = JCS_RGB;
  jpeg_set_defaults(&codec);
  jpeg_set_quality(&codec, 90, TRUE);
  jpeg_start_compress(&codec, TRUE);
  std::vector<uint8_t> row(width * 3);
  while (codec.next_scanline < height) {
    uint32_t y = codec.next_scanline;
    for (uint32_t x = 0; x < width; ++x) {
      // Spatial structure plus deterministic high-frequency detail. This is a
      // synthetic control, never a claimed natural-image workload distribution.
      row[x * 3] = (x * 255 / width + y / 5) & 255;
      row[x * 3 + 1] = ((x / 17 ^ y / 13) & 1) ? 220 : 30;
      row[x * 3 + 2] = (x * 13 + y * 7 + (x * y) % 31) & 255;
    }
    JSAMPROW pointer = row.data();
    if (jpeg_write_scanlines(&codec, &pointer, 1) != 1) {
      Fail("short encode");
    }
  }
  jpeg_finish_compress(&codec);
  jpeg_destroy_compress(&codec);
  if (std::fclose(file)) {
    Fail("source close failed");
  }
}

void Transcode(const Image& image, const std::string& path, bool restart,
               bool progressive) {
  jpeg_decompress_struct decoder{};
  jpeg_compress_struct encoder{};
  jpeg_error_mgr decode_error, encode_error;
  InitializeError(&decode_error);
  InitializeError(&encode_error);
  decoder.err = &decode_error;
  encoder.err = &encode_error;
  jpeg_create_decompress(&decoder);
  jpeg_create_compress(&encoder);
  jpeg_mem_src(&decoder, image.bytes.data(), image.bytes.size());
  jpeg_read_header(&decoder, TRUE);
  auto coefficients = jpeg_read_coefficients(&decoder);
  jpeg_copy_critical_parameters(&decoder, &encoder);
  encoder.restart_interval = restart ? 64 : 0;
  encoder.optimize_coding = TRUE;
  if (progressive) {
    jpeg_simple_progression(&encoder);
  }
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) {
    Fail("cannot create transcode");
  }
  jpeg_stdio_dest(&encoder, file);
  jpeg_write_coefficients(&encoder, coefficients);
  jpeg_finish_compress(&encoder);
  jpeg_finish_decompress(&decoder);
  jpeg_destroy_compress(&encoder);
  jpeg_destroy_decompress(&decoder);
  if (std::fclose(file)) {
    Fail("transcode close failed");
  }
}

void Verify(const Image& image, const std::vector<uint8_t>& storage,
            const std::vector<uint8_t>& reference) {
  if (std::memcmp(storage.data() + 32, reference.data(), image.pixel_bytes)) {
    Fail("full RGB mismatch");
  }
  for (size_t i = 0; i < 32; ++i) {
    if (storage[i] != 0xa5 || storage[32 + image.pixel_bytes + i] != 0xa5) {
      Fail("output guard overwritten");
    }
  }
}

struct Batch;
struct Job {
  // Executor-owned scheduling storage until release_fn's last access.
  iree_task_process_t process;
  // Borrowed immutable input for this operation.
  const Image* image;
  // Distinct guarded destination, retained through full batch retirement.
  std::vector<uint8_t> storage;
  // Batch lifetime exceeds executor retirement and the final completion wait.
  Batch* batch;
};

struct Batch {
  // Completion predicate and last worker publication share this lock.
  std::mutex mutex;
  // One final wait, outside per-image work.
  std::condition_variable condition;
  // Number of processes whose final release callback has not published.
  size_t remaining = 0;
};

iree_status_t Drain(iree_task_process_t* process,
                    const iree_task_worker_context_t*,
                    iree_task_process_drain_result_t* result) {
  auto* job = static_cast<Job*>(process->user_data);
  Decode(*job->image, job->storage.data() + 32);
  result->did_work = true;
  result->completed = true;
  return iree_ok_status();
}

void Release(iree_task_process_t* process) {
  auto* batch = static_cast<Job*>(process->user_data)->batch;
  std::lock_guard<std::mutex> lock(batch->mutex);
  if (--batch->remaining == 0) {
    batch->condition.notify_one();
  }
}

void Run(iree_task_executor_t* executor, Batch& batch,
         std::vector<std::unique_ptr<Job>>& jobs) {
  if (!executor) {
    for (auto& job : jobs) {
      Decode(*job->image, job->storage.data() + 32);
    }
    return;
  }
  batch.remaining = jobs.size();
  for (auto& job : jobs) {
    iree_task_process_initialize(Drain, 0, 1, &job->process);
    job->process.user_data = job.get();
    job->process.completion_fn = [](iree_task_process_t*,
                                    iree_status_t status) { Check(status); };
    job->process.release_fn = Release;
    iree_task_executor_schedule_process(executor, &job->process);
  }
  // A burst can publish all immediate processes before the first worker leaves
  // its idle state. Publish the known batch parallelism once after the queue
  // is populated, instead of treating per-process wake(1) as a batch fan-out.
  iree_task_executor_wake_workers(
      executor, static_cast<int32_t>(std::min(
                    jobs.size(), iree_task_executor_worker_count(executor))));
  std::unique_lock<std::mutex> lock(batch.mutex);
  batch.condition.wait(lock, [&] { return batch.remaining == 0; });
}

void Benchmark(const std::vector<Image>& images, size_t count, size_t workers,
               size_t repetitions, bool timed = true) {
  Batch batch;
  std::vector<std::unique_ptr<Job>> jobs;
  std::vector<std::vector<uint8_t>> references;
  for (const auto& image : images) {
    references.emplace_back(image.pixel_bytes);
    Decode(image, references.back().data());
  }
  size_t input_bytes = 0, output_bytes = 0;
  for (size_t i = 0; i < count; ++i) {
    auto job = std::make_unique<Job>();
    job->image = &images[i % images.size()];
    job->storage.assign(job->image->pixel_bytes + 64, 0xa5);
    job->batch = &batch;
    input_bytes += job->image->bytes.size();
    output_bytes += job->image->pixel_bytes;
    jobs.push_back(std::move(job));
  }
  iree_task_executor_t* executor = nullptr;
  if (workers) {
    iree_task_topology_t topology;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) {
      Fail("cannot query worker placement");
    }
    std::vector<uint32_t> processors;
    for (uint32_t i = 0; i < CPU_SETSIZE && processors.size() < workers; ++i) {
      if (CPU_ISSET(i, &allowed)) {
        processors.push_back(i);
      }
    }
    if (processors.size() != workers) {
      Fail("not enough allowed CPUs for worker placement");
    }
    Check(iree_task_topology_initialize_from_logical_cpu_set(
        processors.size(), processors.data(), &topology));
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    options.worker_local_memory_size = 0;
    Check(iree_task_executor_create(options, &topology, iree_allocator_system(),
                                    &executor));
    iree_task_topology_deinitialize(&topology);
  }
  auto verify = [&] {
    for (size_t i = 0; i < jobs.size(); ++i) {
      Verify(*jobs[i]->image, jobs[i]->storage, references[i % images.size()]);
    }
  };
  Run(executor, batch, jobs);
  verify();
  Run(executor, batch, jobs);
  std::vector<int64_t> samples;
  for (size_t i = 0; i < repetitions; ++i) {
    auto start = std::chrono::steady_clock::now();
    Run(executor, batch, jobs);
    auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count());
  }
  verify();
  if (executor) {
    iree_task_executor_release(executor);
  }
  std::printf(
      "{\"kind\":\"%s\",\"workers\":%zu,\"batch\":%zu,"
      "\"compressed_bytes\":%zu,\"rgb_bytes\":%zu,\"images\":[",
      timed ? "timing" : "qualification_batch", workers, count, input_bytes,
      output_bytes);
  for (size_t i = 0; i < images.size(); ++i) {
    std::printf("%s\"%s\"", i ? "," : "", images[i].name.c_str());
  }
  std::printf("],\"batch_ns\":[");
  for (size_t i = 0; timed && i < samples.size(); ++i) {
    std::printf("%s%lld", i ? "," : "", (long long)samples[i]);
  }
  std::printf("],\"full_rgb_and_guards\":true}\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    Fail(
        "usage: prepare directory images... | qualify images... | bench batch "
        "workers repetitions images...");
  }
  std::string mode = argv[1];
  if (mode == "prepare" || mode == "check") {
    const char* output_directory =
        mode == "check" ? std::getenv("TEST_UNDECLARED_OUTPUTS_DIR") : argv[2];
    if (!output_directory) {
      Fail("check mode requires the Bazel test output directory");
    }
    std::string directory = output_directory;
    std::filesystem::create_directories(directory);
    std::vector<std::string> paths;
    for (int i = mode == "check" ? 2 : 3; i < argc; ++i) {
      paths.emplace_back(argv[i]);
    }
    for (auto shape : {std::pair{256u, 256u}, std::pair{1024u, 768u}}) {
      std::string path =
          directory + "/synthetic-" + std::to_string(shape.first) + ".jpg";
      WriteSynthetic(path, shape.first, shape.second);
      paths.push_back(path);
    }
    for (const auto& path : paths) {
      Image source = ReadImage(path.c_str());
      std::vector<uint8_t> expected(source.pixel_bytes);
      Decode(source, expected.data());
      auto coefficients = Coefficients(source);
      std::vector<Image> variants;
      for (int kind = 0; kind < 3; ++kind) {
        std::string output = directory + "/" + source.name +
                             (kind == 0   ? ".r0.jpg"
                              : kind == 1 ? ".r64.jpg"
                                          : ".progressive.jpg");
        Transcode(source, output, kind == 1, kind == 2);
        Image image = ReadImage(output.c_str());
        std::vector<uint8_t> actual(image.pixel_bytes + 64, 0xa5);
        Decode(image, actual.data() + 32);
        Verify(image, actual, expected);
        if (Coefficients(image) != coefficients) {
          Fail("transcode changed coefficients or quantization");
        }
        std::printf(
            "{\"kind\":\"prepared\",\"name\":\"%s\",\"source\":\"%s\","
            "\"compressed_bytes\":%zu,\"rgb_hash\":\"%016llx\",\"coefficient_"
            "hash\":\"%016llx\",\"full_rgb_equal\":true,\"full_coefficients_"
            "equal\":true}\n",
            image.name.c_str(), source.name.c_str(), image.bytes.size(),
            (unsigned long long)Hash(expected.data(), expected.size()),
            (unsigned long long)Hash(
                reinterpret_cast<const uint8_t*>(coefficients.data()),
                coefficients.size() * 2));
        variants.push_back(std::move(image));
      }
      if (mode == "check") {
        for (size_t workers : {0u, 1u, 4u}) {
          Benchmark(variants, 6, workers, 2, false);
        }
      }
    }
    return 0;
  }
  int start = mode == "bench" ? 5 : 2;
  if (argc <= start) {
    Fail("missing input images");
  }
  std::vector<Image> images;
  for (int i = start; i < argc; ++i) {
    images.push_back(ReadImage(argv[i]));
  }
  if (mode == "qualify") {
    for (const auto& image : images) {
      std::vector<uint8_t> reference(image.pixel_bytes);
      std::vector<uint8_t> storage(image.pixel_bytes + 64, 0xa5);
      Decode(image, reference.data());
      Decode(image, storage.data() + 32);
      Verify(image, storage, reference);
      std::printf(
          "{\"kind\":\"qualified\",\"name\":\"%s\",\"width\":%u,\"height\":%u,"
          "\"compressed_bytes\":%zu,\"rgb_bytes\":%zu,\"progressive\":%s,"
          "\"restart_interval\":%u,\"rgb_hash\":\"%016llx\",\"sampling\":[",
          image.name.c_str(), image.width, image.height, image.bytes.size(),
          image.pixel_bytes, image.progressive ? "true" : "false",
          image.restart_interval,
          (unsigned long long)Hash(reference.data(), reference.size()));
      for (size_t i = 0; i < image.sampling.size(); ++i) {
        std::printf("%s%d", i ? "," : "", image.sampling[i]);
      }
      std::printf("]}\n");
    }
  } else if (mode == "bench") {
    size_t count = std::stoul(argv[2]);
    size_t workers = std::stoul(argv[3]);
    size_t repetitions = std::stoul(argv[4]);
    if (!count || count > 128 || workers > 96 || !repetitions ||
        repetitions > 100) {
      Fail("benchmark arguments exceed experiment bounds");
    }
    Benchmark(images, count, workers, repetitions);
  } else {
    Fail("unknown mode");
  }
  return 0;
}
