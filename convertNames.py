import re 

def convertNames(name):
    name = re.sub("aenm", "assay_component_endpoint_name", name)
    name = re.sub("acnm", "assay_component_name", name)
    name = re.sub("anm", "assay_name", name)
    name = re.sub("asnm", "assay_source_name", name) 
    return name
