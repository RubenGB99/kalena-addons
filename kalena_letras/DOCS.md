# Kalena Letras (fase de prueba)

Pone con IA el **momento exacto de cada palabra** a las letras que ya tienes en Jellyfin, para que
Kalena las muestre como karaoke. No toca tus archivos de música: pide cada canción a Jellyfin y
guarda la letra nueva en Jellyfin, igual que el editor de letras de Kalena (y siempre se puede
volver a la original desde Kalena: «Editar letra» → «Recuperar original»).

## Cómo funciona

1. Separa la voz de la música y, dentro de la voz, **la voz principal de los coros**.
2. Coloca cada palabra de la letra sobre la voz principal (las partes entre paréntesis, que suelen
   ser coros, sobre los coros).
3. Ajusta el comienzo de cada palabra al instante en que arranca la voz.
4. Si una línea no queda clara, **la deja como estaba** en lugar de inventarse tiempos.

## Puesta en marcha

1. **Clave de API de Jellyfin**: en Jellyfin, *Panel de control → Claves de API → +*, ponle de
   nombre «Kalena Letras» y copia la clave.
2. En la pestaña **Configuración** de este complemento:
   - **Dirección de Jellyfin**: deja `http://172.30.32.1:8096` si Jellyfin es un complemento de
     este Home Assistant. Si no conecta, prueba con la IP del mini PC, por ejemplo
     `http://192.168.1.50:8096`.
   - **Clave de API**: la del paso 1 (solo se guarda en Home Assistant).
   - **Usuario**: el usuario de Jellyfin con el que se entra en Kalena. Solo se buscan canciones en
     las bibliotecas que ve ese usuario, y la copia de la letra original queda en su Kalena.
   - **Canciones**: 3 o 4 canciones de estilos distintos, una por línea, como
     `DANNA - SUEÑO MOJADITO`. Si alguna no tiene letra en Jellyfin, se toma de LRCLIB (como
     hace Kalena); se puede desactivar con «Buscar la letra en LRCLIB». Si alguna no la encuentra,
     el registro dice lo más parecido que ve ese usuario, con su id; también se puede poner el id
     directamente (sale al final de la dirección de la canción en la web de Jellyfin, `id=…`).
   - **Separación de voces**: `maxima` (recomendada para la prueba).
3. **Iniciar** el complemento y mirar la pestaña **Registro**.

La primera vez Home Assistant construye el complemento (unos 10 minutos) y descarga los modelos de
IA (unos 2,5 GB, una sola vez; no entran en las copias de seguridad). Con un Intel N100, cada
canción de 3-4 minutos tarda **unos 10-15 minutos**, algo más si tiene coros entre paréntesis.

## Resultados

- La letra con tiempos por palabra queda en Jellyfin y se ve en Kalena.
- En la carpeta compartida `share/kalena_letras` quedan, por cada canción, el `.lrc` y un `.json`
  con el detalle (confianza de cada línea, tiempos de cada paso). Ese `.json` es el que hay que
  mandar para ajustar la precisión.
- Las líneas que la IA no tiene claras se quedan exactamente como estaban: con su tiempo de línea
  o, si ya los tenían, con sus tiempos por palabra. El `.json` dice el motivo de cada una.
- Cuando termina, el registro dice «Terminado» y el complemento se detiene solo (así libera la
  memoria de los modelos).
