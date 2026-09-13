"""Embed and merge standard XMP packets without re-encoding JPEG image data."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree


JPEG_SOI = b"\xff\xd8"
XMP_APP1_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
EXTENDED_XMP_APP1_HEADER = b"http://ns.adobe.com/xmp/extension/\x00"
EXIF_APP1_HEADER = b"Exif\x00\x00"

APP0_MARKER = 0xE0
APP1_MARKER = 0xE1
APP11_MARKER = 0xEB
SOS_MARKER = 0xDA
EOI_MARKER = 0xD9

MAX_APP_MARKER_DATA = 65_533
MAX_STANDARD_XMP_BYTES = MAX_APP_MARKER_DATA - len(XMP_APP1_HEADER)

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NOTE_NS = "http://ns.adobe.com/xmp/note/"
RDF_DESCRIPTION = f"{{{RDF_NS}}}Description"
RDF_ABOUT = f"{{{RDF_NS}}}about"
RDF_ID = f"{{{RDF_NS}}}ID"
RDF_NODE_ID = f"{{{RDF_NS}}}nodeID"
DESCRIPTION_ID_ATTRIBUTES = (RDF_ABOUT, RDF_ID, RDF_NODE_ID)

# UUID of a C2PA manifest-store JUMBF superbox.
C2PA_MANIFEST_UUID = bytes.fromhex("6332706100110010800000aa00389b71")


class JpegXmpError(Exception):
    """Base error raised for a JPEG/XMP contract violation."""


class InvalidJpegError(JpegXmpError):
    """The source file is not a supported, structurally valid JPEG."""


class InvalidXmpError(JpegXmpError):
    """An XMP packet is malformed or lacks an RDF data model."""


class UnsupportedXmpError(JpegXmpError):
    """The source requires Extended XMP or another unsupported feature."""


class ContentCredentialsError(JpegXmpError):
    """The JPEG appears to contain a C2PA Content Credentials manifest."""


@dataclass(frozen=True)
class HeaderSegment:
    """One complete JPEG header segment before the first scan."""

    marker: int
    raw: bytes
    payload: bytes


@dataclass(frozen=True)
class ParsedJpeg:
    """JPEG header segments and the untouched scan-data suffix."""

    segments: tuple[HeaderSegment, ...]
    scan_data: bytes


@dataclass(frozen=True)
class EmbedPlan:
    """Prepared output and facts used by the caller's preflight report."""

    output_data: bytes
    embedded_xmp: bytes
    changed: bool
    had_embedded_xmp: bool


@dataclass(frozen=True)
class WriteResult:
    """Result of an atomic write to one JPEG."""

    changed: bool
    backup_path: Path | None


def parse_jpeg(data: bytes) -> ParsedJpeg:
    """Parse JPEG markers up to SOS and keep the compressed suffix opaque."""

    if len(data) < 4 or not data.startswith(JPEG_SOI):
        raise InvalidJpegError("Файл не начинается с JPEG-маркера SOI.")

    segments: list[HeaderSegment] = []
    position = len(JPEG_SOI)

    while position < len(data):
        marker_start = position
        if data[position] != 0xFF:
            raise InvalidJpegError(
                f"Ожидался JPEG-маркер по смещению {position}."
            )

        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            raise InvalidJpegError("JPEG обрывается внутри маркера.")

        marker = data[position]
        position += 1
        if marker == 0x00:
            raise InvalidJpegError("До начала изображения найден stuffed-маркер FF00.")
        if marker == EOI_MARKER:
            raise InvalidJpegError("JPEG завершился до первого сегмента изображения.")

        # TEM and restart markers do not carry a two-byte segment length.
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            segments.append(
                HeaderSegment(marker, data[marker_start:position], b"")
            )
            continue

        if position + 2 > len(data):
            raise InvalidJpegError("JPEG обрывается перед длиной сегмента.")
        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2:
            raise InvalidJpegError("JPEG содержит сегмент с недопустимой длиной.")
        segment_end = position + segment_length
        if segment_end > len(data):
            raise InvalidJpegError("JPEG обрывается внутри сегмента.")

        if marker == SOS_MARKER:
            return ParsedJpeg(tuple(segments), data[marker_start:])

        payload = data[position + 2 : segment_end]
        segments.append(
            HeaderSegment(marker, data[marker_start:segment_end], payload)
        )
        position = segment_end

    raise InvalidJpegError("В JPEG не найден маркер начала изображения SOS.")


def extract_standard_xmp(data: bytes) -> bytes | None:
    """Return the single embedded standard XMP packet, if present."""

    parsed = parse_jpeg(data)
    packets = [
        segment.payload[len(XMP_APP1_HEADER) :]
        for segment in parsed.segments
        if _is_standard_xmp(segment)
    ]
    if len(packets) > 1:
        raise UnsupportedXmpError(
            "JPEG содержит несколько стандартных XMP-сегментов."
        )
    return packets[0] if packets else None


def merge_xmp_packets(embedded_xmp: bytes, sidecar_xmp: bytes) -> bytes:
    """Merge RDF properties, with sidecar values winning on name conflicts."""

    embedded_root = _parse_xmp_root(embedded_xmp, "встроенный XMP")
    sidecar_root = _parse_xmp_root(sidecar_xmp, "внешний XMP")
    embedded_rdf = _find_rdf(embedded_root, "встроенный XMP")
    sidecar_rdf = _find_rdf(sidecar_root, "внешний XMP")

    destination_descriptions = [
        child for child in embedded_rdf if child.tag == RDF_DESCRIPTION
    ]
    for source_description in sidecar_rdf:
        if source_description.tag != RDF_DESCRIPTION:
            continue
        identity = _description_identity(source_description)
        matching_descriptions = [
            item
            for item in destination_descriptions
            if _description_identity(item) == identity
        ]
        if not matching_descriptions:
            destination = copy.deepcopy(source_description)
            embedded_rdf.append(destination)
            destination_descriptions.append(destination)
            continue
        _merge_description(matching_descriptions, source_description)

    merged = etree.tostring(
        embedded_root,
        encoding="utf-8",
        xml_declaration=False,
        pretty_print=False,
    )
    _parse_xmp_root(merged, "объединённый XMP")
    return merged


def build_jpeg_with_xmp(jpeg_data: bytes, sidecar_xmp: bytes) -> EmbedPlan:
    """Build a JPEG with merged XMP while preserving compressed bytes exactly."""

    sidecar_root = _parse_xmp_root(sidecar_xmp, "внешний XMP")
    _reject_extended_reference(sidecar_root, "внешний XMP")
    parsed = parse_jpeg(jpeg_data)

    standard_indexes = [
        index
        for index, segment in enumerate(parsed.segments)
        if _is_standard_xmp(segment)
    ]
    if len(standard_indexes) > 1:
        raise UnsupportedXmpError(
            "JPEG содержит несколько стандартных XMP-сегментов."
        )
    if any(_is_extended_xmp(segment) for segment in parsed.segments):
        raise UnsupportedXmpError(
            "JPEG уже содержит Extended XMP; безопасное слияние не поддерживается."
        )
    if any(_is_c2pa_manifest(segment) for segment in parsed.segments):
        raise ContentCredentialsError(
            "JPEG содержит Content Credentials (C2PA); изменение пропущено."
        )

    embedded_xmp: bytes | None = None
    if standard_indexes:
        segment = parsed.segments[standard_indexes[0]]
        embedded_xmp = segment.payload[len(XMP_APP1_HEADER) :]
        embedded_root = _parse_xmp_root(embedded_xmp, "встроенный XMP")
        _reject_extended_reference(embedded_root, "встроенный XMP")

    if embedded_xmp == sidecar_xmp:
        merged_xmp = embedded_xmp
    elif embedded_xmp is None:
        merged_xmp = sidecar_xmp
    else:
        merged_xmp = merge_xmp_packets(embedded_xmp, sidecar_xmp)

    if len(merged_xmp) > MAX_STANDARD_XMP_BYTES:
        raise UnsupportedXmpError(
            "Объединённый XMP занимает "
            f"{len(merged_xmp)} байт при лимите {MAX_STANDARD_XMP_BYTES}."
        )
    new_segment = _make_standard_xmp_segment(merged_xmp)

    if embedded_xmp == merged_xmp:
        return EmbedPlan(jpeg_data, merged_xmp, False, True)

    output_segments: list[bytes] = []
    if standard_indexes:
        replace_index = standard_indexes[0]
        for index, segment in enumerate(parsed.segments):
            output_segments.append(new_segment if index == replace_index else segment.raw)
    else:
        insert_index = _default_xmp_insert_index(parsed.segments)
        for index, segment in enumerate(parsed.segments):
            if index == insert_index:
                output_segments.append(new_segment)
            output_segments.append(segment.raw)
        if insert_index == len(parsed.segments):
            output_segments.append(new_segment)

    output_data = JPEG_SOI + b"".join(output_segments) + parsed.scan_data
    verified = parse_jpeg(output_data)
    if verified.scan_data != parsed.scan_data:
        raise InvalidJpegError("Внутренняя проверка обнаружила изменение JPEG-данных.")
    if extract_standard_xmp(output_data) != merged_xmp:
        raise InvalidJpegError("Внутренняя проверка не смогла прочитать записанный XMP.")

    return EmbedPlan(output_data, merged_xmp, True, embedded_xmp is not None)


def write_jpeg_atomically(
    image_path: Path,
    output_data: bytes,
    *,
    backup_original: bool,
    expected_source: bytes | None = None,
) -> WriteResult:
    """Write a verified JPEG through a same-directory temporary file."""

    original_data = image_path.read_bytes()
    if expected_source is not None and original_data != expected_source:
        raise OSError(
            "JPEG изменился после подготовки результата; запись отменена."
        )
    if original_data == output_data:
        return WriteResult(False, None)

    # Re-validate immediately before any filesystem mutation.
    expected_xmp = extract_standard_xmp(output_data)
    if expected_xmp is None:
        raise InvalidJpegError("Подготовленный JPEG не содержит XMP.")
    original_scan = parse_jpeg(original_data).scan_data
    output_scan = parse_jpeg(output_data).scan_data
    if original_scan != output_scan:
        raise InvalidJpegError("Подготовленный JPEG изменяет сжатые данные изображения.")

    backup_path: Path | None = None
    if backup_original:
        backup_path = _ensure_backup(image_path, original_data)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{image_path.name}.",
        suffix=".tmp",
        dir=image_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(output_data)
            stream.flush()
            os.fsync(stream.fileno())
        written_data = temporary_path.read_bytes()
        if written_data != output_data:
            raise OSError("Проверка временного JPEG после записи не пройдена.")
        if extract_standard_xmp(written_data) != expected_xmp:
            raise OSError("XMP во временном JPEG не прошёл проверку.")
        if image_path.read_bytes() != original_data:
            raise OSError(
                "JPEG изменился во время подготовки временного файла; запись отменена."
            )
        os.replace(temporary_path, image_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return WriteResult(True, backup_path)


def _ensure_backup(image_path: Path, original_data: bytes) -> Path:
    """Создать один проверенный backup через временный файл в той же папке."""

    backup_directory = image_path.parent / "backup"
    backup_path = backup_directory / image_path.name
    if backup_directory.exists() and not backup_directory.is_dir():
        raise OSError(
            f"Путь папки резервных копий занят не папкой: {backup_directory}"
        )
    if backup_path.exists():
        if not backup_path.is_file():
            raise OSError(f"Путь резервной копии занят не файлом: {backup_path}")
        return backup_path

    backup_directory.mkdir(exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{image_path.name}.",
        suffix=".backup.tmp",
        dir=backup_directory,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(image_path, temporary_path)
        if temporary_path.read_bytes() != original_data:
            raise OSError("Проверка временной резервной копии не пройдена.")
        os.replace(temporary_path, backup_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return backup_path


def _is_standard_xmp(segment: HeaderSegment) -> bool:
    return segment.marker == APP1_MARKER and segment.payload.startswith(
        XMP_APP1_HEADER
    )


def _is_extended_xmp(segment: HeaderSegment) -> bool:
    return segment.marker == APP1_MARKER and segment.payload.startswith(
        EXTENDED_XMP_APP1_HEADER
    )


def _is_c2pa_manifest(segment: HeaderSegment) -> bool:
    if segment.marker != APP11_MARKER:
        return False
    payload_lower = segment.payload.lower()
    return C2PA_MANIFEST_UUID in segment.payload or (
        b"c2pa" in payload_lower and b"jumb" in payload_lower
    )


def _make_standard_xmp_segment(xmp_data: bytes) -> bytes:
    payload = XMP_APP1_HEADER + xmp_data
    segment_length = len(payload) + 2
    return b"\xff\xe1" + segment_length.to_bytes(2, "big") + payload


def _default_xmp_insert_index(segments: tuple[HeaderSegment, ...]) -> int:
    """Place XMP after JFIF/EXIF metadata without reordering other segments."""

    insert_index = 0
    for index, segment in enumerate(segments):
        if segment.marker == APP0_MARKER or (
            segment.marker == APP1_MARKER
            and segment.payload.startswith(EXIF_APP1_HEADER)
        ):
            insert_index = index + 1
    return insert_index


def _parse_xmp_root(xmp_data: bytes, source_name: str) -> etree._Element:
    if not xmp_data.strip():
        raise InvalidXmpError(f"{source_name} пуст.")
    if b"<!doctype" in xmp_data.lower():
        raise InvalidXmpError(f"{source_name} содержит запрещённый DOCTYPE.")
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        recover=False,
        remove_blank_text=False,
        huge_tree=False,
    )
    try:
        root = etree.fromstring(xmp_data, parser=parser)
    except (etree.XMLSyntaxError, ValueError) as error:
        raise InvalidXmpError(f"{source_name} содержит некорректный XML: {error}") from error
    _find_rdf(root, source_name)
    return root


def _find_rdf(root: etree._Element, source_name: str) -> etree._Element:
    rdf_tag = f"{{{RDF_NS}}}RDF"
    if root.tag == rdf_tag:
        return root
    rdf = root.find(f".//{rdf_tag}")
    if rdf is None:
        raise InvalidXmpError(f"{source_name} не содержит rdf:RDF.")
    return rdf


def _reject_extended_reference(root: etree._Element, source_name: str) -> None:
    extended_tag = f"{{{XMP_NOTE_NS}}}HasExtendedXMP"
    for element in root.iter():
        if element.tag == extended_tag or extended_tag in element.attrib:
            raise UnsupportedXmpError(
                f"{source_name} ссылается на Extended XMP."
            )


def _description_identity(description: etree._Element) -> tuple[str, str]:
    for attribute in DESCRIPTION_ID_ATTRIBUTES:
        if attribute in description.attrib:
            return attribute, description.attrib[attribute]
    return RDF_ABOUT, ""


def _merge_description(
    destinations: list[etree._Element],
    source: etree._Element,
) -> None:
    """Replace matching RDF properties while preserving destination-only data."""

    destination = destinations[0]
    identity_attributes = set(DESCRIPTION_ID_ATTRIBUTES)
    for attribute, value in source.attrib.items():
        if attribute in identity_attributes:
            continue
        for item in destinations:
            for child in list(item):
                if isinstance(child.tag, str) and child.tag == attribute:
                    item.remove(child)
            if item is not destination:
                item.attrib.pop(attribute, None)
        destination.set(attribute, value)

    for source_property in source:
        if not isinstance(source_property.tag, str):
            continue
        for item in destinations:
            item.attrib.pop(source_property.tag, None)
            for destination_property in list(item):
                if destination_property.tag == source_property.tag:
                    item.remove(destination_property)
        destination.append(copy.deepcopy(source_property))
