# Cambios

## 0.1.5
- Solo se apuntan como hechas las canciones cuya letra se ha guardado de verdad en Jellyfin. Las
  que se saltan (sin letra, ya con tiempos, sin alinear o con error) se vuelven a mirar en el
  siguiente arranque, también las que versiones anteriores apuntaron por error.

## 0.1.4
- Canción no encontrada: el registro lo dice claro, «no encontrada en las bibliotecas de <usuario>»,
  porque solo se busca en las bibliotecas que ve el usuario configurado.

## 0.1.3
- Títulos con apóstrofo: «Don't» y «Don’t» se consideran iguales al buscar (Jellyfin los distingue).

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
