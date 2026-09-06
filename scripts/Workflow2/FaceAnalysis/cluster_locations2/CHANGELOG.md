# История изменений cluster_locations2

## v2.2026.09.06

### Исправлено

- При чтении PNG chunk `iCCP` удаляется из копии файла в памяти перед передачей
  в OpenCV. Это устраняет предупреждение libpng
  `iCCP: known incorrect sRGB profile`, не изменяя исходные изображения.
