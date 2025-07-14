'''Load in 47 Tuc Region with LSDB/HATS'''

import lsdb
import numpy as np
from upath import UPath

lsdb.__version__ #Want '0.6.0' . Use the LSDB v0.6 kernel.

base_path = UPath("/epyc/data3/hats/catalogs/dp1")

object_cat = lsdb.open_catalog(base_path / "object_collection") # Use this.
dia_object_cat = lsdb.open_catalog(base_path / "dia_object_collection") # For time-variable stuff (mostly)

tuc = object_cat.cone_search(ra=6.0230792, dec=-72.0814444, radius_arcsec=18000) 
tuc.compute() # This is small enough for the compute operation, but generally avoid this. It tends to revert to lazy-load format after additional operations.