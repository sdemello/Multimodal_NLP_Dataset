import zipfile

with zipfile.ZipFile("ResearchProjectHDF5Files.zip", "r") as zip_ref:
    zip_ref.extractall("./ResearchProjectHDF5Files/")

