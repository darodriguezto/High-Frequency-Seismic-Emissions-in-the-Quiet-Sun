import os
import importlib
import drive_utils.upload
importlib.reload(drive_utils.upload)
from drive_utils.auth import autenticar_drive
from drive_utils.upload import subir_o_actualizar_archivo


service = autenticar_drive()
subir_o_actualizar_archivo(service, "Correlating images from HMI & DKIST.ipynb")
