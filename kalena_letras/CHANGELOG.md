# Cambios

## 0.1.2
- Encuentra mejor las canciones: si no sale con el título tal cual, la busca sin tildes ni signos
  y por el artista.
- Si aun así no la encuentra, el registro dice lo más parecido que ve el usuario (con su id) o que
  ese usuario no ve ninguna parecida (falta de acceso a esa biblioteca).

## 0.1.1
- Se instala bien: el Supervisor construía el complemento sobre su imagen Alpine (sin apt-get) y
  fallaba al instalar ffmpeg. Ahora usa siempre su imagen de Python sobre Debian.

## 0.1.0
- Fase de prueba: separación de voz y coros, alineación palabra por palabra y guardado en Jellyfin
  con copia de la letra original para Kalena.
