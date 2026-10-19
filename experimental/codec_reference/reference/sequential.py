"""Independent T.81 sequential Huffman syntax model for bounded experiments.

This is cold experiment instrumentation, not the serving decoder or CPU
preprocessing for a GPU implementation. Annexes B, C, E, and F define the
format. No third-party decoder implementation is used by this module.
"""

from array import array
from dataclasses import dataclass, field

# Zigzag index -> row-major position in an 8 x 8 coefficient block.
ZIGZAG = (
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
)


class InvalidJPEG(ValueError):
    """External bytes violate a supported format's syntax or bounds."""


class UnsupportedJPEG(ValueError):
    """The input requests a process outside the admitted experiment."""


def require(condition, message):
    if not condition:
        raise InvalidJPEG(message)


def word(data, position):
    require(position + 2 <= len(data), "truncated 16-bit field")
    return data[position] * 256 + data[position + 1]


def divide_up(value, divisor):
    return (value + divisor - 1) // divisor


@dataclass
class Huffman:
    # Canonical code lookup, indexed by (1 << length) | code.
    symbols: dict

    @classmethod
    def create(cls, counts, values, kind):
        require(len(counts) == 16 and sum(counts) == len(values), "DHT extent")
        require(0 < len(values) <= 256, "DHT symbol count")
        # Codewords identify decoder entries. Distinct codewords may map to
        # the same symbol; only an encoder's inverse lookup needs uniqueness.
        symbols = {}
        code = 0
        index = 0
        for length, count in enumerate(counts, 1):
            # JPEG reserves the all-ones code, including at the deepest level.
            require(code + count < (1 << length), "overfull/all-ones Huffman tree")
            for _ in range(count):
                value = values[index]
                if kind == 0:
                    require(value <= 11, "DC category exceeds 8-bit DCT range")
                else:
                    require(
                        value in (0, 0xF0) or 1 <= (value & 15) <= 10,
                        "AC category is not 8-bit sequential DCT",
                    )
                symbols[(1 << length) | code] = value
                code += 1
                index += 1
            code <<= 1
        return cls(symbols)

    def decode(self, reader):
        reader.huffman_attempts += 1
        code = 1
        for _ in range(16):
            code = (code << 1) | reader.take(1)
            value = self.symbols.get(code)
            if value is not None:
                reader.huffman_symbols += 1
                return value
        raise InvalidJPEG("undefined Huffman code")


@dataclass
class Component:
    # Header identifier, independent of colorspace interpretation.
    identifier: int
    # Number of horizontally interleaved blocks in each MCU.
    horizontal: int
    # Number of vertically interleaved blocks in each MCU.
    vertical: int
    # Quantization table destination from the frame header.
    quantization_id: int
    # Actual component width in blocks, excluding interleaved dummy blocks.
    block_width: int = 0
    # Actual component height in blocks, excluding interleaved dummy blocks.
    block_height: int = 0
    # Natural-order table frozen when this component's scan starts.
    quantization: tuple = ()


@dataclass
class Block:
    # Index into the frame's components, also the DC predictor index.
    component: int
    # Horizontal block coordinate within the MCU's component rectangle.
    column: int
    # Vertical block coordinate within the MCU's component rectangle.
    row: int
    # Immutable DC table captured at this scan's SOS.
    dc: Huffman
    # Immutable AC table captured at this scan's SOS.
    ac: Huffman


@dataclass
class Segment:
    # Unstuffed entropy bytes, ending before a restart or scan-ending marker.
    data: bytes
    # First encoded byte in the original compressed allocation.
    source_start: int
    # Encoded byte count, including stuffing but excluding the ending marker.
    source_bytes: int
    # Zero-based absolute MCU index within this scan.
    first_mcu: int
    # Exact MCU count before a restart or the scan's end.
    mcu_count: int


@dataclass
class Scan:
    # Blocks in interleaved order; one block for a noninterleaved scan.
    blocks: list
    # Number of MCUs in one scan row.
    mcu_width: int
    # Interleaved scans tile each component by its sampling rectangle.
    interleaved: bool
    # Independently anchored restart segments in encoded order.
    segments: list


@dataclass
class Frame:
    # Encoded maximum horizontal sample count.
    width: int
    # Encoded maximum vertical sample count.
    height: int
    # SOF0 or SOF1 marker code.
    process: int
    # Component geometry in frame header order.
    components: list
    # Sequential scans in encoded order, each component appearing once.
    scans: list = field(default_factory=list)


def split_entropy(data, start):
    """Return unstuffed segments and the position of the scan-ending marker."""
    position = start
    segment_start = start
    chunks = []
    segments = []
    restart = 0
    while True:
        marker_start = data.find(b"\xff", position)
        require(marker_start >= 0, "entropy has no ending marker")
        chunks.append(data[position:marker_start])
        position = marker_start + 1
        require(position < len(data), "truncated entropy marker")
        if data[position] == 0:
            chunks.append(b"\xff")
            position += 1
            continue
        while position < len(data) and data[position] == 255:
            position += 1
        require(position < len(data), "truncated fill marker")
        marker = data[position]
        require(marker != 0, "fill bytes before stuffed entropy byte")
        segments.append((b"".join(chunks), segment_start, marker_start - segment_start))
        if not 0xD0 <= marker <= 0xD7:
            return segments, marker_start
        require(marker == 0xD0 + restart, "restart sequence mismatch")
        restart = (restart + 1) % 8
        position += 1
        segment_start = position
        chunks = []


def parse(data):
    """Validate 8-bit sequential Huffman framing and freeze scan descriptors."""
    require(data[:2] == b"\xff\xd8", "missing SOI")
    position = 2
    quantization = {}
    huffman = {}
    restart_interval = 0
    frame = None
    completed = set()
    while position < len(data):
        require(data[position] == 255, "missing marker prefix")
        while position < len(data) and data[position] == 255:
            position += 1
        require(position < len(data), "truncated marker")
        marker = data[position]
        position += 1
        if marker == 0xD9:
            require(frame is not None and frame.scans, "EOI without image")
            require(len(completed) == len(frame.components), "missing component scan")
            require(position == len(data), "bytes after EOI in experiment input")
            return frame
        require(
            marker not in (0, 1, 0xD8, *range(0xD0, 0xD8)),
            "unexpected standalone marker",
        )
        length = word(data, position)
        require(length >= 2 and position + length <= len(data), "marker length")
        payload = data[position + 2 : position + length]
        position += length
        if marker in (
            *range(0xC0, 0xC4),
            *range(0xC5, 0xC8),
            *range(0xC9, 0xCC),
            *range(0xCD, 0xD0),
        ):
            if marker not in (0xC0, 0xC1):
                raise UnsupportedJPEG(
                    f"SOF {marker:02x} outside sequential Huffman experiment"
                )
            require(frame is None, "multiple frames")
            require(len(payload) >= 6, "short frame header")
            if payload[0] != 8:
                raise UnsupportedJPEG(
                    f"precision {payload[0]} outside 8-bit experiment"
                )
            height, width, count = word(payload, 1), word(payload, 3), payload[5]
            require(
                width != 0 and count != 0 and len(payload) == 6 + count * 3,
                "frame dimensions/component extent",
            )
            if height == 0:
                raise UnsupportedJPEG("DNL height is outside this bounded experiment")
            components = []
            for offset in range(6, len(payload), 3):
                identifier, sampling, table = payload[offset : offset + 3]
                horizontal, vertical = sampling >> 4, sampling & 15
                require(
                    1 <= horizontal <= 4 and 1 <= vertical <= 4 and table <= 3,
                    "component sampling/quantization selector",
                )
                require(
                    identifier not in [c.identifier for c in components],
                    "duplicate frame component",
                )
                components.append(Component(identifier, horizontal, vertical, table))
            maximum_h = max(c.horizontal for c in components)
            maximum_v = max(c.vertical for c in components)
            for component in components:
                component.block_width = divide_up(
                    width * component.horizontal, 8 * maximum_h
                )
                component.block_height = divide_up(
                    height * component.vertical, 8 * maximum_v
                )
            frame = Frame(width, height, marker, components)
        elif marker == 0xDB:
            offset = 0
            require(payload, "empty DQT")
            while offset < len(payload):
                descriptor = payload[offset]
                offset += 1
                precision, table = descriptor >> 4, descriptor & 15
                require(precision <= 1 and table <= 3, "DQT precision/destination")
                width = precision + 1
                require(offset + 64 * width <= len(payload), "truncated DQT")
                values = [0] * 64
                for index in range(64):
                    value = payload[offset] if width == 1 else word(payload, offset)
                    require(value != 0, "zero quantization entry")
                    values[ZIGZAG[index]] = value
                    offset += width
                quantization[table] = tuple(values)
        elif marker == 0xC4:
            offset = 0
            require(payload, "empty DHT")
            while offset < len(payload):
                require(offset + 17 <= len(payload), "truncated DHT counts")
                descriptor = payload[offset]
                kind, table = descriptor >> 4, descriptor & 15
                require(kind <= 1 and table <= 3, "DHT class/destination")
                counts = payload[offset + 1 : offset + 17]
                offset += 17
                count = sum(counts)
                require(offset + count <= len(payload), "truncated DHT values")
                huffman[kind, table] = Huffman.create(
                    counts, payload[offset : offset + count], kind
                )
                offset += count
        elif marker == 0xDD:
            require(len(payload) == 2, "DRI extent")
            restart_interval = word(payload, 0)
        elif marker == 0xDA:
            require(frame is not None and payload, "scan without frame/header")
            count = payload[0]
            require(1 <= count <= 4 and len(payload) == 4 + count * 2, "SOS extent")
            require(payload[-3:] == bytes([0, 63, 0]), "nonsequential scan parameters")
            selected = []
            blocks = []
            for offset in range(1, 1 + count * 2, 2):
                identifier, selectors = payload[offset : offset + 2]
                indices = [
                    i
                    for i, c in enumerate(frame.components)
                    if c.identifier == identifier
                ]
                require(len(indices) == 1, "unknown scan component")
                index = indices[0]
                require(
                    index not in completed and (not selected or selected[-1] < index),
                    "repeated/out-of-order scan component",
                )
                selected.append(index)
                component = frame.components[index]
                dc, ac = selectors >> 4, selectors & 15
                limit = 1 if frame.process == 0xC0 else 3
                require(dc <= limit and ac <= limit, "scan Huffman destination")
                require(
                    (0, dc) in huffman and (1, ac) in huffman, "missing Huffman table"
                )
                require(
                    component.quantization_id in quantization,
                    "missing quantization table",
                )
                component.quantization = quantization[component.quantization_id]
                if count == 1:
                    blocks.append(Block(index, 0, 0, huffman[0, dc], huffman[1, ac]))
                else:
                    for row in range(component.vertical):
                        for column in range(component.horizontal):
                            blocks.append(
                                Block(
                                    index, column, row, huffman[0, dc], huffman[1, ac]
                                )
                            )
            require(len(blocks) <= 10, "more than ten blocks per interleaved MCU")
            if count == 1:
                component = frame.components[selected[0]]
                mcu_width, mcu_height = component.block_width, component.block_height
            else:
                maximum_h = max(c.horizontal for c in frame.components)
                maximum_v = max(c.vertical for c in frame.components)
                mcu_width = divide_up(frame.width, 8 * maximum_h)
                mcu_height = divide_up(frame.height, 8 * maximum_v)
            mcu_count = mcu_width * mcu_height
            chunks, position = split_entropy(data, position)
            interval = restart_interval or mcu_count
            require(
                len(chunks) == divide_up(mcu_count, interval),
                "restart count disagrees with DRI",
            )
            segments = [
                Segment(
                    chunk,
                    start,
                    encoded,
                    i * interval,
                    min(interval, mcu_count - i * interval),
                )
                for i, (chunk, start, encoded) in enumerate(chunks)
            ]
            frame.scans.append(Scan(blocks, mcu_width, count > 1, segments))
            completed.update(selected)
        elif marker == 0xFE or 0xE0 <= marker <= 0xEF:
            # Metadata has no effect on quantized coefficient syntax.
            pass
        else:
            raise UnsupportedJPEG(f"marker {marker:02x} outside sequential experiment")
    raise InvalidJPEG("missing EOI")


class Bits:
    def __init__(self, data, position=0, limit=None):
        # Immutable unstuffed entropy bytes for one restart segment.
        self.data = data
        # Next bit position, measured from this segment's first byte.
        self.position = position
        # Exclusive bound, including encoded padding unless specified.
        self.limit = len(data) * 8 if limit is None else limit
        # Codeword decode attempts, including invalid speculative prefixes.
        self.huffman_attempts = 0
        # Successfully recognized codewords, before amplitude/run validation.
        self.huffman_symbols = 0

    def take(self, count):
        end = self.position + count
        require(end <= self.limit, "truncated entropy/amplitude")
        value = 0
        while self.position < end:
            byte, bit = divmod(self.position, 8)
            width = min(8 - bit, end - self.position)
            value = (value << width) | (
                (self.data[byte] >> (8 - bit - width)) & ((1 << width) - 1)
            )
            self.position += width
        return value

    def amplitude(self, count):
        if count == 0:
            return 0
        value = self.take(count)
        return value if value >= (1 << (count - 1)) else value - (1 << count) + 1


def step(reader, block, coefficient):
    """One syntax transition: (next coefficient phase, written index, value).

    Phase zero starts DC. Returning zero completes the current block. Index
    -1 means an EOB or ZRL with no nonzero coefficient; DC values are deltas.
    Predictor accumulation and absolute block placement belong to the caller.
    """
    if coefficient == 0:
        category = block.dc.decode(reader)
        return 1, 0, reader.amplitude(category)
    symbol = block.ac.decode(reader)
    run, category = symbol >> 4, symbol & 15
    if category == 0:
        if run == 0:
            return 0, -1, 0
        coefficient += 16
        require(coefficient <= 64, "ZRL exceeds block")
        return coefficient % 64, -1, 0
    index = coefficient + run
    require(index < 64, "AC run exceeds block")
    return (index + 1) % 64, index, reader.amplitude(category)


@dataclass
class Trace:
    # Symbol-start bit offsets plus one terminal boundary; segment-relative.
    positions: array = field(default_factory=lambda: array("I"))
    # (Block-in-MCU << 6) | coefficient phase at each corresponding boundary.
    phases: array = field(default_factory=lambda: array("H"))
    # Absolute completed block count within this segment at every boundary.
    blocks: array = field(default_factory=lambda: array("I"))


def destination(frame, scan, mcu, block):
    """Natural-order base index; None for an encoded edge dummy block."""
    component = frame.components[block.component]
    mcu_row, mcu_column = divmod(mcu, scan.mcu_width)
    if scan.interleaved:
        row = mcu_row * component.vertical + block.row
        column = mcu_column * component.horizontal + block.column
    else:
        row, column = mcu_row, mcu_column
    if row >= component.block_height or column >= component.block_width:
        return None
    return (row * component.block_width + column) * 64


def decode(frame, record_trace=False):
    """Exact signed quantized planes, with optional compact syntax traces."""
    planes = [
        array("h", [0]) * (c.block_width * c.block_height * 64)
        for c in frame.components
    ]
    traces = []
    for scan in frame.scans:
        for segment in scan.segments:
            reader = Bits(segment.data)
            predictors = [0] * len(frame.components)
            trace = Trace() if record_trace else None
            block_count = segment.mcu_count * len(scan.blocks)
            for absolute_block in range(block_count):
                mcu, block_index = divmod(absolute_block, len(scan.blocks))
                block = scan.blocks[block_index]
                base = destination(frame, scan, segment.first_mcu + mcu, block)
                plane = planes[block.component]
                coefficient = 0
                while True:
                    if trace is not None:
                        trace.positions.append(reader.position)
                        trace.phases.append(block_index * 64 + coefficient)
                        trace.blocks.append(absolute_block)
                    coefficient, index, value = step(reader, block, coefficient)
                    if index == 0:
                        predictors[block.component] += value
                        value = predictors[block.component]
                        require(-1024 <= value <= 1023, "DC outside 8-bit DCT range")
                    if base is not None and index >= 0:
                        plane[base + ZIGZAG[index]] = value
                    if coefficient == 0:
                        break
            if trace is not None:
                trace.positions.append(reader.position)
                trace.phases.append(0)
                trace.blocks.append(block_count)
                traces.append(trace)
            padding = reader.limit - reader.position
            require(0 <= padding < 8, "entropy remains after last MCU")
            require(
                reader.take(padding) == (1 << padding) - 1, "non-ones entropy padding"
            )
    return planes, traces
