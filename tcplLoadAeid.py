from query_db import tcplQuery
from buildAssayQ import buildAssayQ

def tcplLoadAeid(fld=None, val=None, add_fld=None):
    out = ["assay_component_endpoint.aeid","assay_component_endpoint.assay_component_endpoint_name"]
    qstring = buildAssayQ(out=out, tblo=[0, 1, 3, 2, 5], fld=fld, val=val, add_fld=add_fld)
    dat = tcplQuery(query=qstring)
    return dat
