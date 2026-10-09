# Cambios

## 0.1.1
- Se instala bien: el Supervisor construía el complemento sobre su imagen Alpine (sin apt-get) y
  fallaba al instalar ffmpeg. Ahora usa siempre su imagen de Python sobre Debian.

## 0.1.0
- Fase de prueba: separación de voz y coros, alineación palabra por palabra y guardado en Jellyfin
  con copia de la letra original para Kalena.
