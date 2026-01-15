import os
from googleapiclient.http import MediaFileUpload
from .config import FOLDER_ID_TESIS


def subir_o_actualizar_archivo(service, ruta_local, folder_id=FOLDER_ID_TESIS):
    """
    Sube un archivo a Google Drive.
    Si ya existe en la carpeta, actualiza su contenido.
    Si no existe, lo crea.
    """

    if not os.path.exists(ruta_local):
        raise FileNotFoundError(f"No existe el archivo local: {ruta_local}")

    nombre = os.path.basename(ruta_local)

    query = (
        f"name='{nombre}' and "
        f"'{folder_id}' in parents and "
        "trashed=false"
    )

    resultados = service.files().list(
        q=query,
        fields="files(id, name)"
    ).execute()

    archivos = resultados.get("files", [])

    media = MediaFileUpload(ruta_local, resumable=True)

    if archivos:
        file_id = archivos[0]["id"]
        service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        print(f"🔄 Archivo actualizado en Drive: {nombre}")

    else:
        metadata = {
            "name": nombre,
            "parents": [folder_id]
        }
        service.files().create(
            body=metadata,
            media_body=media,
            fields="id"
        ).execute()
        print(f"⬆️ Archivo subido a Drive: {nombre}")

