"""Idioma de la interfaz: español (el del código) o inglés.

t(texto) devuelve el texto en el idioma elegido. Las claves de EN son los textos en español tal y como
salen en el programa; "{}" marca una parte variable (un número, un nombre...), que también se traduce
si es una frase conocida. Lo que no esté en la tabla se queda en español.
"""

from __future__ import annotations

import json
import locale
import re

from . import paths

LANGS = {"es": "Español", "en": "English"}


def _initial_lang() -> str:
    try:
        lang = json.loads((paths.CONFIG_DIR / "ajustes.json").read_text(encoding="utf-8")).get("idioma")
        if lang in LANGS:
            return lang
    except Exception:  # noqa: BLE001
        pass
    loc = (locale.getlocale()[0] or "").lower()
    return "es" if loc.startswith(("es", "spanish")) else "en"


LANG = _initial_lang()

EN = {
    # ---------------------------------------------------------------- ventana principal
    "En cola": "Queued",
    "Procesando": "Processing",
    "En pausa": "Paused",
    "Terminada": "Done",
    "Error": "Error",
    "Cancelada": "Cancelled",
    "{} h {} min": "{} h {} min",
    "Arrastra aquí tus canciones (mp3, flac, wav, m4a...)\no pulsa «Añadir canciones» o «Añadir de YouTube»":
        "Drag your songs here (mp3, flac, wav, m4a...)\nor press \"Add songs\" or \"Add from YouTube\"",
    "Charter {} — charts para YARG y Clone Hero": "Charter {} — charts for YARG and Clone Hero",
    "Convierte tus canciones en charts jugables para YARG / Clone Hero: guitarra, rítmica, bajo, batería, "
    "teclado, Pro Keys y voz, en Fácil, Medio, Difícil y Experto.":
        "Turns your songs into playable charts for YARG / Clone Hero: guitar, rhythm, bass, drums, keys, "
        "Pro Keys and vocals, in Easy, Medium, Hard and Expert.",
    "Añadir canciones...": "Add songs...",
    "Añadir de YouTube...": "Add from YouTube...",
    "Descarga la música de un enlace de YouTube y, si quieres, también su vídeo para el fondo.":
        "Downloads the music from a YouTube link and, if you want, its video for the background.",
    "Quitar": "Remove",
    "Canción seleccionada": "Selected song",
    "Sin\nportada": "No\ncover",
    "Cambiar portada...": "Change cover...",
    "Video de fondo...": "Background video...",
    "Sin video": "No video",
    "Abrir visualizador": "Open viewer",
    "Abrir carpeta": "Open folder",
    "Regenerar": "Regenerate",
    "Vuelve a crear el chart con lo último aprendido. Reutiliza la separación y la transcripción ya hechas, "
    "así que tarda poco.":
        "Makes the chart again with the latest training. It reuses the separation and transcription already "
        "done, so it's quick.",
    "Carpeta de salida": "Output folder",
    "Elegir...": "Choose...",
    "Proceso": "Progress",
    "Listo para empezar": "Ready to start",
    "Generar charts": "Generate charts",
    "Pausar": "Pause",
    "Cancelar canción": "Cancel song",
    "Entrenar con mi biblioteca de canciones...": "Train with my song library...",
    "Aprende de tus charts (por ejemplo tu carpeta de Clone Hero / YARG) cómo tienen que ser las canciones. "
    "Puedes dejarlo varias horas; solo lee tus archivos.":
        "Learns from your charts (for example your Clone Hero / YARG folder) what songs should look like. "
        "You can leave it running for hours; it only reads your files.",
    "Abrir en el visualizador un chart ya generado...": "Open an existing chart in the viewer...",
    "Consola (lo que está haciendo el programa en cada momento)": "Console (what the program is doing right now)",
    "Limpiar": "Clear",
    "Charter {} listo. Carpeta del programa: {}": "Charter {} ready. Program folder: {}",
    "Añadir canciones": "Add songs",
    "Elegir portada": "Choose cover",
    "Imagen (*.png *.jpg *.jpeg *.webp *.bmp)": "Image (*.png *.jpg *.jpeg *.webp *.bmp)",
    "Cancelar": "Cancel",
    "¿Cancelar la canción que se está procesando?": "Cancel the song that is being processed?",
    "Regenerando {} con lo último aprendido...": "Regenerating {} with the latest training...",
    "Carpeta de la canción (con notes.mid)": "Song folder (with notes.mid)",
    "Entrenando el modelo de dificultades con tus charts (puede tardar)...":
        "Training the difficulty model with your charts (this can take a while)...",
    "Canción": "Song",
    "Estado": "Status",
    "Progreso": "Progress",
    "Tiempo restante": "Time left",
    "Título": "Title",
    "Artista": "Artist",
    "Álbum": "Album",
    "Año": "Year",
    "Género": "Genre",
    "automático": "automatic",
    "Añadida: {}": "Added: {}",
    "No se puede quitar la canción que se está procesando. Cancélala primero.":
        "You can't remove the song that is being processed. Cancel it first.",
    "Sin portada\n(se buscará\nen internet)": "No cover\n(it will be\nsearched online)",
    "Video: {}\nDesfase: {} ms": "Video: {}\nOffset: {} ms",
    "Añade primero alguna canción.": "Add a song first.",
    "En pausa — la tarjeta gráfica está libre. Pulsa «Reanudar» para seguir.":
        "Paused — the graphics card is free. Press \"Resume\" to continue.",
    "Reanudar": "Resume",
    "Esta canción: {}   ·   Toda la cola: {}": "This song: {}   ·   Whole queue: {}",
    "Cola terminada: {} listas": "Queue finished: {} done",
    ", {} con error": ", {} with errors",
    "Visualizador": "Viewer",
    "En esa carpeta no hay un notes.mid.": "There is no notes.mid in that folder.",
    "Entrenar": "Train",
    "Se están generando canciones. Entrenar a la vez hará que las dos cosas vayan más lentas. ¿Abrir igualmente?":
        "Songs are being generated. Training at the same time will make both slower. Open anyway?",
    "Entrenamiento terminado: las próximas canciones usarán lo aprendido.":
        "Training finished: the next songs will use what it learned.",
    "Salir": "Quit",
    "Hay canciones en proceso. Si cierras se cancelará todo lo que no haya terminado. ¿Salir?":
        "Songs are being processed. If you close, everything that hasn't finished will be cancelled. Quit?",
    "Error inesperado": "Unexpected error",
    "{}\n\nDetalles guardados en config/errores.log": "{}\n\nDetails saved in config/errores.log",
    "Modelo de dificultades entrenado con tu biblioteca.": "Difficulty model trained with your library.",
    "Tarjeta grafica: {} ({} GB). Se usara para la IA.": "Graphics card: {} ({} GB). It will be used for the AI.",
    "No se ha encontrado una tarjeta NVIDIA: todo ira por CPU (mucho mas lento).":
        "No NVIDIA card found: everything will run on the CPU (much slower).",
    "No se pudo comprobar la GPU: {}": "Couldn't check the GPU: {}",
    "No se pudo entrenar el modelo de dificultades: {}": "Couldn't train the difficulty model: {}",
    "Idioma": "Language",
    "Reinicia Charter para cambiar el idioma.": "Restart Charter to change the language.",
    # ---------------------------------------------------------------- cola de trabajos
    "Reanudando (se recargan los modelos, puede tardar unos segundos)...":
        "Resuming (the models are reloaded, this can take a few seconds)...",
    "No se pudo lanzar el proceso: {}": "Couldn't start the process: {}",
    "===== Empezando: {} =====": "===== Starting: {} =====",
    "===== Reanudando: {} =====": "===== Resuming: {} =====",
    "Pausa: cerrando el proceso para liberar la tarjeta grafica...":
        "Pause: closing the process to free the graphics card...",
    "El proceso termino de forma inesperada (codigo {})": "The process ended unexpectedly (code {})",
    "Listo: {}": "Done: {}",
    "AVISO: {}": "WARNING: {}",
    "ERROR: {}": "ERROR: {}",
    # ---------------------------------------------------------------- entrenar
    "Empezando": "Starting",
    "Entrenar con mi biblioteca de canciones": "Train with my song library",
    "Charter aprende de tus charts cómo tienen que ser las canciones: cuántas notas, cuántos acordes, HOPOs, "
    "Star Power y cómo se simplifican en Difícil, Medio y Fácil (vale para todas las canciones, aunque solo "
    "tengan chart). Con las que traen las pistas separadas (guitar.ogg, bass.ogg...) aprende además qué golpes "
    "del audio se convierten en nota y cuándo poner acordes.\n\nSolo se leen tus archivos, no se modifica nada. "
    "Lo analizado se guarda: si lo paras, la próxima vez sigue donde lo dejó.":
        "Charter learns from your charts what songs should look like: how many notes, how many chords, HOPOs, "
        "Star Power and how they are simplified in Hard, Medium and Easy (this works for every song, even if it "
        "only has a chart). From the ones that come with separate tracks (guitar.ogg, bass.ogg...) it also "
        "learns which hits in the audio become notes and when to use chords.\n\nIt only reads your files, "
        "nothing is changed. What it analyses is saved: if you stop it, next time it continues where it left off.",
    "Carpeta de tus canciones (por ejemplo la de YARG / Clone Hero)": "Your songs folder (for example YARG / Clone Hero's)",
    "Sin límite (hasta que lo analice todo; puede tardar días)": "No limit (until everything is analysed; it can take days)",
    "Aprovechar todo el tiempo con la IA de separación: separa las canciones sin pistas y vuelve a separar las "
    "que sí las traen, para aprender con el mismo audio que verá al generar (usa la tarjeta gráfica, ~1-2 min "
    "por canción)":
        "Use all the time with the separation AI: separates the songs without tracks and separates again the "
        "ones that have them, to learn from the same audio it will see when generating (uses the graphics card, "
        "~1-2 min per song)",
    "Recomendado. Sin esta opción, cuando acaba con las canciones que traen pistas pasa directamente a entrenar "
    "aunque quede tiempo.":
        "Recommended. Without this option, once it finishes the songs that have tracks it goes straight to "
        "training, even if there is time left.",
    "Listo": "Ready",
    "Volver a lo de serie": "Reset to default",
    "Borra lo aprendido de tu biblioteca y vuelve a los modelos que trae el programa":
        "Deletes what it learned from your library and goes back to the program's own models",
    "Empezar a entrenar": "Start training",
    "Terminar ya y entrenar": "Finish now and train",
    "Deja de analizar canciones y entrena los modelos con todo lo analizado hasta ahora":
        "Stops analysing songs and trains the models with everything analysed so far",
    "Parar": "Stop",
    "Para del todo sin entrenar (lo analizado se guarda para la próxima vez)":
        "Stops completely without training (what was analysed is kept for next time)",
    "Carpeta de tus canciones": "Your songs folder",
    "Entrenamiento iniciado": "Training started",
    "{}  ·  tiempo máximo restante: {} h {} min": "{}  ·  max time left: {} h {} min",
    "{}  ·  sin límite · lleva {} h {} min": "{}  ·  no limit · running for {} h {} min",
    "Terminando: acaba la canción que está analizando y entrena con todo lo analizado...":
        "Finishing: completes the song it is analysing and trains with everything analysed...",
    "Parado. Lo analizado queda guardado: la próxima vez seguirá desde aquí.":
        "Stopped. What was analysed is saved: next time it will continue from here.",
    "¿Borrar lo aprendido de tu biblioteca?": "Delete what it learned from your library?",
    "Se han restaurado los modelos de serie (lo analizado de tu biblioteca sigue guardado).":
        "The default models are back (what was analysed from your library is still saved).",
    "Tiempo máximo (horas):": "Max time (hours):",
    "Elige la carpeta de tus canciones.": "Choose your songs folder.",
    "Entrenando": "Training",
    "El entrenamiento sigue en marcha. ¿Pararlo y cerrar?": "Training is still running. Stop it and close?",
    "Terminado": "Finished",
    # ---------------------------------------------------------------- vídeo
    "Video de fondo": "Background video",
    "Pega aqui un enlace de YouTube...": "Paste a YouTube link here...",
    "Descargar y ver": "Download and view",
    "Elegir archivo de video...": "Choose video file...",
    "Elige un video o pega un enlace de YouTube (se descarga sin audio, como maximo 1080p).":
        "Choose a video or paste a YouTube link (downloaded without audio, up to 1080p).",
    "▶  Reproducir juntos": "▶  Play together",
    "Positivo = el video va por delante (se salta el principio del video). Negativo = el video empieza mas "
    "tarde. Ajusta hasta que lo que ves coincida con lo que oyes.":
        "Positive = the video goes ahead (the start of the video is skipped). Negative = the video starts "
        "later. Adjust until what you see matches what you hear.",
    "Aceptar": "OK",
    "Elegir video": "Choose video",
    "Descargando video...": "Downloading video...",
    "Video: {}": "Video: {}",
    "Desfase del video (ms):": "Video offset (ms):",
    "⏸  Pausa": "⏸  Pause",
    "No se pudo descargar: {}": "Couldn't download: {}",
    # ---------------------------------------------------------------- YouTube
    "Añadir de YouTube": "Add from YouTube",
    "Pega aquí uno o varios enlaces de YouTube, uno por línea...": "Paste one or more YouTube links here, one per line...",
    "Música y vídeo (de fondo en el juego)": "Music and video (as the in-game background)",
    "Solo la música": "Music only",
    "Añadir a las descargas": "Add to downloads",
    "Se descargan de una en una y cada canción pasa a la lista al terminar. Puedes añadir más enlaces mientras "
    "tanto, y luego cambiar o quitar el vídeo con «Video de fondo...».":
        "They download one at a time and each song goes to the list when it's done. You can add more links in "
        "the meantime, and later change or remove the video with \"Background video...\".",
    "Cerrar": "Close",
    "Cancelar descargas": "Cancel downloads",
    "Descargando {} de {}...": "Downloading {} of {}...",
    "Añadida": "Added",
    "Enlace": "Link",
    "Descargar": "Download",
    "En espera": "Waiting",
    "Error: {}": "Error: {}",
    "Cancelando...": "Cancelling...",
    "Descargas terminadas: {} añadidas a la lista": "Downloads finished: {} added to the list",
    "Música y vídeo": "Music and video",
    ", {} con error (pon el ratón encima para ver el motivo).": ", {} with errors (hover to see why).",
    # ---------------------------------------------------------------- visualizador
    "Guitarra": "Guitar",
    "Rítmica": "Rhythm",
    "Bajo": "Bass",
    "Batería": "Drums",
    "Teclado": "Keys",
    "Voz": "Vocals",
    "Resto": "Rest",
    "Público": "Crowd",
    "RACHA": "STREAK",
    "Visualizador - {}": "Viewer - {}",
    "Rápida": "Fast",
    "Normal": "Normal",
    "Lenta": "Slow",
    "▶  Reproducir": "▶  Play",
    "Espacio: reproducir/pausar · Rueda del ratón: avanzar/retroceder · Clic en una nota y Supr: quitarla":
        "Space: play/pause · Mouse wheel: forward/back · Click a note and press Delete: remove it",
    "Resumen del chart": "Chart summary",
    "Guardar cambios": "Save changes",
    "Sin chart cargado": "No chart loaded",
    "Instrumento:": "Instrument:",
    "Dificultad:": "Difficulty:",
    "Velocidad de notas:": "Note speed:",
    "Pistas:": "Tracks:",
    "Batería: {} golpes en Experto": "Drums: {} hits on Expert",
    "Voz: {} notas": "Vocals: {} notes",
    "Guardado": "Saved",
    "Los cambios se han guardado en notes.mid": "The changes were saved in notes.mid",
    "Cambios sin guardar": "Unsaved changes",
    "¿Guardar los cambios antes de cerrar?": "Save the changes before closing?",
    "{}: {} notas en Experto ({}/s)": "{}: {} notes on Expert ({}/s)",
    "No se pudo guardar: {}": "Couldn't save: {}",
    "Fácil": "Easy",
    "Medio": "Medium",
    "Difícil": "Hard",
    "Experto": "Expert",
    # ---------------------------------------------------------------- consola al generar
    "Preparando audio": "Preparing audio",
    "Metadatos y portada": "Metadata and cover",
    "Separando instrumentos (IA)": "Separating instruments (AI)",
    "Detectando tempo y compases": "Detecting tempo and bars",
    "Transcribiendo notas (IA)": "Transcribing notes (AI)",
    "Voz y letra": "Vocals and lyrics",
    "Generando el chart": "Generating the chart",
    "Exportando audio": "Exporting audio",
    "--- {} ---": "--- {} ---",
    "Pulsos ajustados a las notas, poco a poco: error medio {} -> {} ms":
        "Beats adjusted to the notes, gradually: average error {} -> {} ms",
    "Descargando modelo de pulso {}...": "Downloading beat model {}...",
    "Pulsos ajustados a las notas: {} ms (error medio {} -> {} ms)": "Beats adjusted to the notes: {} ms (average error {} -> {} ms)",
    "transcribiendo la letra con Whisper large-v3...": "transcribing the lyrics with Whisper large-v3...",
    "letra alineada: {} palabras en {} lineas ({})": "lyrics aligned: {} words in {} lines ({})",
    "escuchando la voz con Whisper (palabra a palabra)...": "listening to the vocals with Whisper (word by word)...",
    "se oyen {} palabras": "{} words heard",
    "letra sincronizada con lo que se oye: {} lineas ({} de sus palabras se oyen{})":
        "lyrics synced with what is heard: {} lines ({} of their words are heard{})",
    ", {} lineas no se oyen en este audio y se quitan": ", {} lines aren't heard in this audio and are removed",
    "{} lineas de la letra de internet no se cantan en este audio: se quitan":
        "{} lines of the online lyrics aren't sung in this audio: removed",
    "{} lineas recolocadas con los tiempos de la letra sincronizada": "{} lines moved using the synced lyrics timing",
    "solo se oye el {} de la letra de internet: se encaja en toda la cancion":
        "only {} of the online lyrics is heard: fitted across the whole song",
    "LRCLIB no responde ({}), reintentando...": "LRCLIB isn't responding ({}), retrying...",
    "etiquetas del archivo": "file tags",
    "iTunes no disponible: {}": "iTunes not available: {}",
    "MusicBrainz no disponible: {}": "MusicBrainz not available: {}",
    "Duracion: {}:{}": "Length: {}:{}",
    "Cancion: {} - {} | album: {} | ano: {} | genero: {} | portada: {}":
        "Song: {} - {} | album: {} | year: {} | genre: {} | cover: {}",
    "no encontrada": "not found",
    "Tempo: {} ({} BPM aprox.)": "Tempo: {} (about {} BPM)",
    "fijo {} BPM": "fixed {} BPM",
    "variable": "variable",
    "Terminado: {}": "Finished: {}",
    "drums: nivel {} dB, suena el {}% -> {}": "drums: level {} dB, plays {}% of the time -> {}",
    "keys: nivel {} dB, {} notas por minuto -> {}": "keys: level {} dB, {} notes per minute -> {}",
    "{}: nivel {} dB, suena el {}% del tiempo -> {}{}": "{}: level {} dB, plays {}% of the time -> {}{}",
    "se chartea": "charted",
    "no hay": "none",
    "no hay (no se crea pista)": "none (no track created)",
    " (demasiado flojo: seguramente restos de otros instrumentos)": " (too quiet: probably bleed from other instruments)",
    "dos guitarras distintas detectadas ({}/{} tramos diferentes): principal = canal {}":
        "two different guitars detected ({}/{} parts differ): main = {} channel",
    "izquierdo": "left",
    "derecho": "right",
    "idioma de la voz: {}": "vocals language: {}",
    "analizando el tono de la voz (CREPE)...": "analysing the vocal pitch (CREPE)...",
    "Instrumentos: {}": "Instruments: {}",
    "guitarra": "guitar",
    "bajo": "bass",
    "bateria": "drums",
    "teclado": "keys",
    "voz": "vocals",
    "archivo": "file",
    "usuario": "yours",
    "{}: {} notas votadas -> {} con energia real en la pista": "{}: {} voted notes -> {} with real energy in the track",
    "transcripcion de bateria (YourMT3)": "drum transcription (YourMT3)",
    "No se detecta voz: la cancion es instrumental": "No vocals detected: the song is instrumental",
    "letra no encontrada: se transcribe con Whisper": "lyrics not found: transcribing with Whisper",
    "Se anaden {} s de silencio al principio para que de tiempo a ver las primeras notas":
        "Adding {} s of silence at the start so there's time to see the first notes",
    "Generando {}...": "Generating {}...",
    "{}: {} notas en experto ({}/s, {}% acordes)": "{}: {} notes on expert ({}/s, {}% chords)",
    "Generando bateria...": "Generating drums...",
    "Generando voz...": "Generating vocals...",
    "No hay conexion a internet: no se podra buscar la letra, la portada ni los datos que falten, ni descargar "
    "el video de YouTube. La letra se sacara con IA (Whisper).":
        "No internet connection: lyrics, cover and missing song info can't be looked up, and YouTube videos "
        "can't be downloaded. The lyrics will be made with AI (Whisper).",
    "La cancion no tiene portada y no se ha encontrado en internet. Puedes anadirla tu desde la lista de canciones.":
        "The song has no cover and none was found online. You can add one from the song list.",
    "Buscando la letra en internet (LRCLIB)...": "Looking for the lyrics online (LRCLIB)...",
    "Aviso: no se pudieron detectar las secciones ({})": "Warning: couldn't detect the sections ({})",
    "solos de guitarra: {}": "guitar solos: {}",
    "{}: muy pocas notas, se omite": "{}: too few notes, skipped",
    "bateria (modelo neuronal): {} golpes": "drums (neural model): {} hits",
    "bateria: {} golpes en experto, {} de doble bombo": "drums: {} hits on expert, {} double kick",
    "voz: {} notas en {} frases (letra: {}), {} sin tono": "vocals: {} notes in {} phrases (lyrics: {}), {} unpitched",
    "Aviso: no se pudo guardar la portada ({})": "Warning: couldn't save the cover ({})",
    "No se pudo preparar el video: {}": "Couldn't prepare the video: {}",
    "letra encontrada ({}, {} lineas)": "lyrics found ({}, {} lines)",
    "sincronizada": "synced",
    "sin tiempos": "no timing",
    "no se pudo comparar la letra con lo que se oye ({}); se usa entera":
        "couldn't compare the lyrics with what is heard ({}); using all of it",
    "pro keys: {} golpes en experto": "pro keys: {} hits on expert",
    "Separando instrumentos (BS-RoFormer SW, pasada 1: mezcla completa)...":
        "Separating instruments (BS-RoFormer SW, pass 1: full mix)...",
    "Separando instrumentos (pasada 2: mezcla sin voz)...": "Separating instruments (pass 2: mix without vocals)...",
    "Separando voz principal de los coros...": "Separating lead vocals from backing vocals...",
    "Separando la bateria por piezas (bombo, caja, toms, charles, ride, crash)...":
        "Separating the drums by piece (kick, snare, toms, hi-hat, ride, crash)...",
    "Separacion ya hecha (reutilizando)": "Separation already done (reusing)",
    "Separando voz con {}...": "Separating vocals with {}...",
    "(reutilizando resultado guardado de {})": "(reusing saved result from {})",
    "Aviso: no se pudo separar la voz principal ({}); se usa la voz completa":
        "Warning: couldn't separate the lead vocals ({}); using the full vocals",
    "transcripcion pasada 1 (mezcla mono)": "transcription pass 1 (mono mix)",
    "transcripcion pasada 2 (canal izquierdo)": "transcription pass 2 (left channel)",
    "transcripcion pasada 3 (canal derecho)": "transcription pass 3 (right channel)",
    "transcripcion pasada 4 (desplazada)": "transcription pass 4 (shifted)",
    "Descargando video de YouTube: {}": "Downloading YouTube video: {}",
    "Video descargado: {} ({}p)": "Video downloaded: {} ({}p)",
    "ffprobe no pudo leer el video: {}": "ffprobe couldn't read the video: {}",
    "El archivo no tiene imagen de video": "The file has no video image",
    "Convirtiendo el video a H.264 para que el juego pueda reproducirlo...":
        "Converting the video to H.264 so the game can play it...",
    "ffmpeg no pudo preparar el video: {}": "ffmpeg couldn't prepare the video: {}",
    "Buscando el video...": "Looking for the video...",
    "Ese enlace es una lista de reproduccion: pega el enlace de una sola cancion.":
        "That link is a playlist: paste the link of a single song.",
    "Ese video es un directo que sigue en marcha: no se puede descargar entero.":
        "That video is a live stream that is still going: it can't be downloaded in full.",
    "Descargando la musica...": "Downloading the music...",
    "Descargando el video...": "Downloading the video...",
    "pulso: Beat This (3 modelos)": "beats: Beat This (3 models)",
    "transcripcion: YourMT3+": "transcription: YourMT3+",
    "letra: Whisper large-v3": "lyrics: Whisper large-v3",
    "separacion: {}": "separation: {}",
    "separacion: no disponible ({})": "separation: not available ({})",
    "fallo: {}": "failed: {}",
    "alineador de letra ({})": "lyrics aligner ({})",
    "ffmpeg no pudo leer el audio: {}": "ffmpeg couldn't read the audio: {}",
    "ffmpeg no pudo crear {}: {}": "ffmpeg couldn't create {}: {}",
    "bateria: {} golpes detectados": "drums: {} hits detected",
    "bateria: densidad ajustada ({} charles quitados)": "drums: density adjusted ({} hi-hats removed)",
    "densidad reducida de {} a {} notas/s (tope ~{})": "density reduced from {} to {} notes/s (cap ~{})",
    "partes con los mismos acordes (aunque cambie el ritmo): {} de la cancion":
        "parts with the same chords (even if the rhythm changes): {} of the song",
    "partes repetidas: {} grupos de pulsos que suenan igual ({} de la cancion es repeticion)":
        "repeated parts: {} groups of beats that sound the same ({} of the song is repeated)",
    "{} notas elegidas por el modelo neuronal": "{} notes chosen by the neural model",
    "{} notas elegidas por el modelo aprendido de charts": "{} notes chosen by the model learned from charts",
    "{} notas quitadas: eran golpes del bajo que se colaban en la guitarra":
        "{} notes removed: they were bass hits bleeding into the guitar",
    "secciones: {}": "sections: {}",
    "Chart generado automaticamente con Charter": "Chart generated automatically with Charter",
    # nombres de las secciones (van dentro del chart)
    "Intro": "Intro",
    "Final": "Outro",
    "Estribillo": "Chorus",
    "Estrofa": "Verse",
    "Puente": "Bridge",
    "Instrumental": "Instrumental",
    # ---------------------------------------------------------------- consola al entrenar
    "Entrenando el modelo neuronal de notas con {} guitarras separadas y {} originales (unos {} min con la GPU) ...":
        "Training the neural note model with {} separated and {} original guitars (about {} min on the GPU) ...",
    "Fase 1: aprendiendo de {} charts (densidad, acordes, HOPOs, dificultades)...":
        "Phase 1: learning from {} charts (density, chords, HOPOs, difficulties)...",
    "estadisticas guardadas ({} canciones, {} oficiales)": "stats saved ({} songs, {} official)",
    "Modelo de dificultades": "Difficulty model",
    "modelos de dificultad listos": "difficulty models ready",
    "Fase final: entrenando los modelos de audio con todo lo analizado...":
        "Final phase: training the audio models with everything analysed...",
    "pistas analizadas: guitarra {} ({} separadas por la IA), bajo {} ({} separadas por la IA)":
        "tracks analysed: guitar {} ({} separated by the AI), bass {} ({} separated by the AI)",
    "modelos de audio guardados": "audio models saved",
    "Guitarras originales para el modelo neuronal (las que traen guitar.ogg)...":
        "Original guitars for the neural model (the ones with guitar.ogg)...",
    "Buscando charts en {} ...": "Looking for charts in {} ...",
    "Encontradas {} canciones con chart": "Found {} songs with a chart",
    "{} canciones apartadas como banco de pruebas (no se usan para entrenar)":
        "{} songs set aside as a test bench (not used for training)",
    "Entrenamiento terminado en {} min. Las proximas canciones usaran lo aprendido.":
        "Training finished in {} min. The next songs will use what it learned.",
    "modelo neuronal: hacen falta mas guitarras separadas con IA ({} de 100)":
        "neural model: more AI-separated guitars are needed ({} of 100)",
    "modelo de dificultades (guitarra/bajo/teclado) ya entrenado con estas canciones: se reutiliza":
        "difficulty model (guitar/bass/keys) already trained with these songs: reusing it",
    "entrenando el modelo de dificultades (guitarra/bajo/teclado) con {} canciones...":
        "training the difficulty model (guitar/bass/keys) with {} songs...",
    "bateria desactivada: no se entrena su modelo de dificultades": "drums turned off: their difficulty model isn't trained",
    "Fase 2: audio. {} canciones traen pistas separadas{}": "Phase 2: audio. {} songs come with separate tracks{}",
    "; despues, con el tiempo que quede, la IA separara las {} que no las traen y volvera a separar las que si "
    "(asi aprende con audio como el que vera al generar)":
        "; then, with the time left, the AI will separate the {} that don't and separate again the ones that do "
        "(so it learns from audio like the one it will see when generating)",
    "muy pocas pistas con audio: se mantienen los modelos de audio de serie":
        "very few tracks with audio: keeping the default audio models",
    "No se han encontrado canciones (carpetas con notes.mid o notes.chart y song.ini)":
        "No songs found (folders with notes.mid or notes.chart and song.ini)",
    "no se pudo entrenar el modelo neuronal: {}": "couldn't train the neural model: {}",
    "separacion rapida inestable: sigo en precision normal": "fast separation unstable: continuing at normal precision",
    "Analizando charts": "Analysing charts",
    "{}/{} charts analizados": "{}/{} charts analysed",
    "modelo de dificultades de bateria ya entrenado: se reutiliza": "drum difficulty model already trained: reusing it",
    "entrenando el modelo de dificultades de bateria...": "training the drum difficulty model...",
    "se acaba el tiempo: paso a entrenar con lo analizado": "time is up: moving on to training with what was analysed",
    "'Terminar ya': paso a entrenar con lo analizado": "'Finish now': moving on to training with what was analysed",
    "Analizando audio": "Analysing audio",
    "{}: pocas pistas ({}); se mantiene el modelo de serie ({} pistas)":
        "{}: few tracks ({}); keeping the default model ({} tracks)",
    "aviso: no se pudieron preparar las guitarras originales ({})": "warning: couldn't prepare the original guitars ({})",
    "{}/{} charts preparados": "{}/{} charts prepared",
    "{} canciones con audio analizadas ({})": "{} songs with audio analysed ({})",
    "aviso: no se pudo guardar el espectrograma de {} ({})": "warning: couldn't save the spectrogram of {} ({})",
    "aviso: {}: {}": "warning: {}: {}",
    "analizada: {}": "analysed: {}",
    "error en {}: {}": "error in {}: {}",
    "{} canciones para entrenar ({} separadas por la IA, {} con la pista original), {} para validar ({} apartadas: estan en el banco)":
        "{} songs for training ({} separated by the AI, {} with the original track), {} for validation ({} set aside: they're in the bench)",
    "{} canciones para entrenar ({} separadas por la IA), {} para validar ({} separadas por la IA)":
        "{} songs for training ({} separated by the AI), {} for validation ({} separated by the AI)",
    "{} canciones para entrenar ({} separadas por la IA, {} Pro), {} para validar ({} separadas)":
        "{} songs for training ({} separated by the AI, {} Pro), {} for validation ({} separated)",
    "modelo de ancho {}: {} M parametros": "width {} model: {} M parameters",
    "botones: acierta si sube, baja o repite en el {} de las notas de validacion":
        "buttons: gets up, down or same right on {} of the validation notes",
    "Modelo neuronal guardado: validacion F1 {} con umbrales {} ({} canciones)":
        "Neural model saved: validation F1 {} with thresholds {} ({} songs)",
    "Hacen falta mas espectrogramas: entrena antes con la biblioteca y separacion con IA.":
        "More spectrograms are needed: train first with the library and AI separation.",
    "se sigue entrenando {}": "continuing to train {}",
    "Modelo neuronal: parado antes de poder medirlo, no se guarda": "Neural model: stopped before it could be measured, not saved",
    "paso {}: perdida {} | validacion F1 {} (umbral {}) | {} min restantes":
        "step {}: loss {} | validation F1 {} (threshold {}) | {} min left",
    "Modelo neuronal: el anterior sigue siendo mejor ({} frente a {}), se queda":
        "Neural model: the previous one is still better ({} vs {}), keeping it",
    "{} canciones sin teclado como ejemplo de lo que no son notas": "{} songs without keys as examples of what isn't a note",
    "teclas de Pro Keys: F1 {} en las notas de validacion (umbral {})": "Pro Keys keys: F1 {} on the validation notes (threshold {})",
    "Modelo de teclado guardado: 5 carriles {}, Pro Keys {}, umbrales {} {} ({} canciones)":
        "Keys model saved: 5 lanes {}, Pro Keys {}, thresholds {} {} ({} songs)",
    "teclas: golpes {} pista, {} separadas, {} validacion": "keys: hits {} track, {} separated, {} validation",
    "teclas: guardada {}": "keys: saved {}",
    "Hacen falta mas espectrogramas de teclado (keys_spectra).": "More keys spectrograms are needed (keys_spectra).",
    "Modelo de teclado: parado antes de poder medirlo, no se guarda": "Keys model: stopped before it could be measured, not saved",
    "teclas: la red anterior sigue siendo mejor ({} frente a {}), se queda":
        "keys: the previous network is still better ({} vs {}), keeping it",
    "paso {}: perdida {} | validacion 5 carriles {} (umbral {}), Pro Keys {} (umbral {}) | {} min restantes":
        "step {}: loss {} | validation 5 lanes {} (threshold {}), Pro Keys {} (threshold {}) | {} min left",
    "Modelo de teclado: el anterior sigue siendo mejor ({} frente a {}), se queda":
        "Keys model: the previous one is still better ({} vs {}), keeping it",
    "teclas, paso {}: perdida {} | {} | {} min restantes": "keys, step {}: loss {} | {} | {} min left",
    "Modelo de bateria guardado: {}": "Drum model saved: {}",
    "Hacen falta mas espectrogramas de bateria (drum_spectra).": "More drum spectrograms are needed (drum_spectra).",
    "Modelo de bateria: el anterior sigue siendo mejor ({} frente a {}), se queda":
        "Drum model: the previous one is still better ({} vs {}), keeping it",
    "paso {}: perdida {} | validacion F1 {} (golpes {}) {} | {} min restantes":
        "step {}: loss {} | validation F1 {} (hits {}) {} | {} min left",
    "{} canciones con bajo en la lista": "{} songs with bass in the list",
    "espectrogramas de bajo: {}": "bass spectrograms: {}",
    "separacion con IA terminada: {} canciones": "AI separation finished: {} songs",
    "{}/{} ({}) ~{} min restantes": "{}/{} ({}) ~{} min left",
    "{} separadas, {} s cada una{}": "{} separated, {} s each{}",
    "{} separadas, {} s cada una": "{} separated, {} s each",
    "{} canciones con bateria en la lista": "{} songs with drums in the list",
    "espectrogramas de bateria: {}": "drum spectrograms: {}",
    "{} canciones en la lista de separacion": "{} songs in the separation list",
    "{} canciones traen la pista de teclado": "{} songs come with the keys track",
    "espectrogramas de teclado: {}": "keys spectrograms: {}",
    " | parecido con el teclado real: piano {}, piano+resto {} ({} canciones)":
        " | similarity to the real keys: piano {}, piano+rest {} ({} songs)",
    "{} canciones traen la pista de guitarra original": "{} songs come with the original guitar track",
    "espectrogramas de pistas originales: {}": "original track spectrograms: {}",
    "{} [{}]: {} ataques, {} son notas (de {})": "{} [{}]: {} onsets, {} are notes (of {})",
    "{}: {} pistas | ataque->nota P={} R={} F1={} (umbral {}, medido en {})":
        "{}: {} tracks | onset->note P={} R={} F1={} (threshold {}, measured on {})",
    "audio separado por la IA": "AI-separated audio",
    "pistas originales": "original tracks",
    "guitar: acierto del tamano de acorde (1/2/3 notas) = {}": "guitar: chord size accuracy (1/2/3 notes) = {}",
    "{}: F1 de notas conservadas (validacion cruzada por cancion) = {}": "{}: F1 of kept notes (cross-validation by song) = {}",
    "{}: acierto de carril = {}": "{}: lane accuracy = {}",
    "bateria {}: F1 golpes conservados (validacion cruzada) = {}": "drums {}: F1 of kept hits (cross-validation) = {}",
    "modelo guardado en {}": "model saved in {}",
}

_PATTERNS = sorted(
    ((re.compile("^" + "(.*?)".join(map(re.escape, k.split("{}"))) + "$", re.S), v) for k, v in EN.items() if "{}" in k),
    key=lambda p: -len(p[0].pattern),
)


def t(text: str) -> str:
    """El texto en el idioma elegido (en español, tal cual)."""
    if LANG == "es" or not isinstance(text, str) or not text.strip():
        return text
    body = text.strip()
    pre, post = text[: len(text) - len(text.lstrip())], text[len(text.rstrip()):]
    out = EN.get(body)
    if out is None:
        for rx, tpl in _PATTERNS:
            m = rx.match(body)
            if m:
                parts = iter(t(g) for g in m.groups())
                out = re.sub(r"\{\}", lambda _: next(parts), tpl)
                break
    if out is None and ", " in body:  # listas como "guitarra, bajo, voz"
        items = body.split(", ")
        if all(i in EN for i in items):
            out = ", ".join(EN[i] for i in items)
    return text if out is None else pre + out + post


def set_lang(lang: str) -> None:
    global LANG
    LANG = lang


if __name__ == "__main__":
    set_lang("en")
    assert t("Pausar") == "Pause"
    assert t("  bateria: 300 golpes en experto, 12 de doble bombo") == "  drums: 300 hits on expert, 12 double kick"
    assert t("guitar: nivel -3 dB, suena el 80% del tiempo -> se chartea") == "guitar: level -3 dB, plays 80% of the time -> charted"
    assert t("Cancion: A - B | album: ? | ano: 2020 | genero: Rock | portada: no encontrada").endswith("cover: not found")
    assert t("Instrumentos: guitarra, bajo, voz") == "Instruments: guitar, bass, vocals"
    assert t("Algo que no está en la tabla") == "Algo que no está en la tabla"
    set_lang("es")
    assert t("Pausar") == "Pausar"
    print("ok")
