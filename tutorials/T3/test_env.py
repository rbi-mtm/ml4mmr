#!/usr/bin/env python3
import os
import traceback
import warnings
from copy import deepcopy
from pathlib import Path

import requests

# Suppress some common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def test_atomate2_basic() -> bool:
    """Test basic Atomate2 functionality."""
    print("\n" + "=" * 60)
    print("TESTING ATOMATE2 BASIC FUNCTIONALITY")
    print("=" * 60)

    try:
        # Test importing key atomate2 modules
        from atomate2.forcefields.jobs import ForceFieldRelaxMaker
        from pymatgen.core import Lattice, Structure

        lattice = Lattice.cubic(4.0)
        structure = Structure(lattice, ["Si"], [[0, 0, 0]])
        relax_maker = ForceFieldRelaxMaker(force_field_name="MACE-MP-0")
        relax_job = relax_maker.make(structure=structure)
        from jobflow.managers.fireworks import flow_to_workflow

        fireworks = flow_to_workflow(relax_job)
        from fireworks import LaunchPad

        lpad = LaunchPad.auto_load()
        lpad.add_wf(fireworks)
    except ImportError as e:
        print(f"✗ Atomate2 test failed: {e}")
        return False
    return True


def test_optimade_queries() -> bool:
    """Test OPTIMADE query functionality.
    adapted from https://github.com/Materials-Consortia/optimade-tutorial-exercises
    """
    try:
        print("\n" + "=" * 60)
        print("TESTING OPTIMADE QUERIES")
        print("=" * 60)

        rest_base_oqmd = "http://oqmd.org/optimade/structures?"
        rest_base_alex = "https://alexandria.icams.rub.de/pbe/v1/structures?"

        filter_oqmd = (
            '_oqmd_stability<=0 AND elements HAS "O"'
            " AND nelements=3 AND _oqmd_band_gap>0"
        )
        filter_alex = (
            '_alexandria_hull_distance<=0 AND elements HAS "O"'
            " AND nelements=3 AND _alexandria_band_gap>0"
        )
        # filter_alex   = 'elements HAS "Pb" AND nelements=1'

        response_oqmd = (
            "id,_oqmd_entry_id,lattice_vectors,cartesian_site_positions,"
            "species_at_sites,_oqmd_band_gap"
        )
        response_alex = (
            "id,chemical_formula_descriptive,lattice_vectors,cartesian_site_positions,"
            "species_at_sites,_alexandria_band_gap"
        )
        page_ = ["page_offset=0", "page_limit=10"]

        filter_oqmd = "filter=" + filter_oqmd
        filter_alex = "filter=" + filter_alex
        response_oqmd = "response_fields=" + response_oqmd
        response_alex = "response_fields=" + response_alex

        oqmd_optimade_query = rest_base_oqmd + "&".join(
            [filter_oqmd, response_oqmd] + page_
        )
        print("Created Query: \n\n{}".format(oqmd_optimade_query))
        alexandria_optimade_query = rest_base_alex + "&".join(
            [filter_alex, response_alex] + page_
        )
        print("Created Query: \n\n{}".format(alexandria_optimade_query))

        def query_optimade(query):
            response = requests.get(query)
            if response.status_code == 200:
                return response.json()
            else:
                print("Query failed. Status: {}".format(response.status_code))
                print("Error Message: {}".format(response.text))
                return None

        dataset_oqmd = []
        for i in range(10):
            jsondata = query_optimade(oqmd_optimade_query)
            if jsondata is None:
                break
            else:
                # Get the link to the next page and query it in next loop iteration
                deepcopy(jsondata["links"]["next"])
                dataset_oqmd.append(deepcopy(jsondata))
        print("Fetched {} pages from OQMD".format(len(dataset_oqmd)))
        print("\n\n")
        dataset_alexandria = []
        for i in range(1):
            jsondata = query_optimade(alexandria_optimade_query)
            if jsondata is None:
                break
            else:
                # Get the link to the next page and query it in next loop iteration
                deepcopy(jsondata["links"]["next"])
                dataset_alexandria.append(deepcopy(jsondata))
        print("Fetched {} pages from Alexandria".format(len(dataset_alexandria)))

    except Exception as e:
        print(f"✗ OPTIMADE query test failed: {e}")
        traceback.print_exc()
        return False

    properties = []
    # temporary directory to store POSCAR files
    poscar_dir = Path("./test_poscars")
    poscar_dir.mkdir(parents=True, exist_ok=True)
    for dt in dataset_oqmd:
        for st in dt["data"]:
            poscar, filename = get_poscar_from_optimade_structure(deepcopy(st))
            target_value = deepcopy(st["attributes"]["_oqmd_band_gap"])
            properties.append(",".join([filename, str(target_value)]))
            with open(os.path.join(poscar_dir, filename), "w") as fout:
                fout.write(poscar)
    print("POSCAR files saved in:", poscar_dir)
    for dt in dataset_alexandria:
        for st in dt["data"]:
            poscar, filename = get_poscar_from_optimade_structure(deepcopy(st))
            target_value = deepcopy(st["attributes"]["_alexandria_band_gap"])
            properties.append(",".join([filename, str(target_value)]))
            with open(os.path.join(poscar_dir, filename), "w") as fout:
                fout.write(poscar)

    return True


def get_poscar_from_optimade_structure(structure):
    if "_oqmd_entry_id" in structure["attributes"].keys():
        poscar = [
            "REST API StructureID {}, OQMD Entry ID {}".format(
                structure["id"], structure["attributes"]["_oqmd_entry_id"]
            )
        ]
        filename = "ID-{}_OQMD-EnID-{}.poscar".format(
            structure["id"], structure["attributes"]["_oqmd_entry_id"]
        )
    else:
        poscar = ["REST API StructureID {}".format(structure["id"])]
        filename = "ID-{}.poscar".format(structure["id"])

    poscar.append("1.0")

    poscar += [
        " ".join([str(jtem) for jtem in item])
        for item in structure["attributes"]["lattice_vectors"]
    ]

    elems = []
    counts = []
    for item in structure["attributes"]["species_at_sites"]:
        if item in elems:
            assert elems.index(item) == len(elems) - 1
            counts[-1] += 1
        else:
            elems.append(deepcopy(item))
            counts.append(1)
    poscar.append(" ".join(elems))
    poscar.append(" ".join([str(item) for item in counts]))

    poscar.append("Cartesian")

    poscar += [
        " ".join([str(jtem) for jtem in item])
        for item in structure["attributes"]["cartesian_site_positions"]
    ]
    poscar = "\n".join(poscar)
    return (poscar, filename)


def run_all_tests() -> None:
    """Run all tests and print results."""

    results = {}

    # Test Atomate2 basic functionality
    results["atomate2"] = test_atomate2_basic()

    # Test OPTIMADE queries
    results["optimade"] = test_optimade_queries()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for test, result in results.items():
        if isinstance(result, dict):
            for package, success in result.items():
                status = "✓" if success else "✗"
                print(f"{test} - {package}: {status}")
        else:
            status = "✓" if result else "✗"
            print(f"{test}: {status}")


if __name__ == "__main__":
    run_all_tests()
