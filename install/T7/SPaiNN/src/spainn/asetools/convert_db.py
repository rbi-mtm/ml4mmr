import logging

import ase.db
import numpy as np

import spainn
from spainn.properties import SPAINN
from spainn.asetools.aseutils import DatabaseUtils

log = logging.getLogger(__name__)


__all__ = ["ConvertDB"]


class ConvertDB(DatabaseUtils):
    """
    Convert your old SchNarc database to a SPaiNN compatible db,
    or add smooth NACs to existing SPaiNN db
    """

    def __init__(self) -> None:
        super().__init__()

    def add_smooth_nacs(
        self, olddb: str, newdb: str = "", copy_metadata: bool = True
    ) -> None:
        """
        Adds smooth NACs to an existing SPaiNN db

        Args:
            olddb:          Path of the old SPaiNN db
            newdb:          Path for the extended db, if not passed
                            "_new" will be appended to the old name
            copy_metadata:  Copy metadata from olddb to newdb
        """
        if newdb == "":
            newdb = self.generate_name(olddb)
        self.checkpaths(olddb, newdb)

        log.info("Adding smooth nacs to %s -> %s", olddb, newdb)

        converted = ase.db.connect(newdb)
        with ase.db.connect(olddb) as conn:
            log.info("%s has %s entries", olddb, len(conn))
            log.info("%s keys: %s", olddb, " ".join(conn.get(1).data.keys()))

            if SPAINN.nacs not in conn.get(1).data.keys():
                raise AttributeError(f"{olddb} does not contain NACs")

            for row in conn.select():
                props = row.data
                props["smooth_nacs"] = self.calc_smooth_nacs(
                    props[SPAINN.nacs], props[SPAINN.energy]
                )
                converted.write(row.toatoms(), data=props)

        # Copy old metadata to new db
        if copy_metadata:
            metadata = conn.metadata
            if "_property_unit_dict" in metadata:
                metadata["_property_unit_dict"]["smooth_nacs"] = "1"
            converted.metadata = metadata

    def convert(
        self,
        olddb: str,
        newdb: str = "",
        copy_metadata: bool = True,
        smooth_nacs: bool = False,
    ) -> None:
        """
        Converts your db

        Args:
            olddb:          Path of the old SchNarc db
            newdb:          Path for the converted db, if not passed
                            "_new" will be appended to the old name
            copy_metadata:  Copy metadata from olddb to newdb
            smooth_nacs:    Adds smooth NACs to new db
        """
        # Construct newdb id not given and check if paths valid
        if newdb == "":
            newdb = self.generate_name(olddb)
        self.checkpaths(olddb, newdb)

        log.info("Converting %s into %s", olddb, newdb)

        # Connect to new db
        converted = ase.db.connect(newdb)

        # Iterate over old db entries, convert and write to new db
        with ase.db.connect(olddb) as conn:
            log.info("%s has %s entries", olddb, len(conn))
            log.info("%s keys: %s", olddb, " ".join(conn.get(1).data.keys()))

            if SPAINN.nacs not in conn.get(1).data.keys() and smooth_nacs:
                log.warning("Smooth NACs are requested, but db does not contain NACs")

            # Main loop, conversion
            for row in conn.select():
                props = row.data
                new_props = {}
                for key, val in props.items():
                    if key == SPAINN.energy:
                        new_props[key] = val.reshape(1, -1)

                    elif key in (SPAINN.forces, SPAINN.nacs):
                        new_props[key] = np.einsum("ijk->jik", val)
                        if key == SPAINN.nacs and smooth_nacs:
                            new_props["smooth_nacs"] = self.calc_smooth_nacs(
                                new_props[SPAINN.nacs], props[SPAINN.energy]
                            )
                    elif key in (SPAINN.dipoles, SPAINN.socs):
                        new_props[key] = val


                converted.write(row.toatoms(), data=new_props)

            # Copy old metadata to new db
            if copy_metadata:
                metadata = conn.metadata
                if smooth_nacs and "_property_unit_dict" in metadata:
                    metadata["_property_unit_dict"]["smooth_nacs"] = "1"
                converted.metadata = metadata
