"""Inspect compiled MuJoCo body masses and inertias for the Diogenes model.

Run from the repo root WITH the STL meshes present:
    python -m diogenes_mjlab.tools.check_masses

It compiles the real scene.xml (meshes included) and reports, per body:
  * mass (kg)
  * diagonalized inertia (principal moments, kg*m^2)
  * the FULL 3x3 inertia tensor in the body frame (kg*m^2)
  * whether the inertia is physically valid (positive moments + triangle ineq.)
and flags any body with ~zero or missing mass.

Why a reconstruction is needed for the 3x3: MuJoCo does not store the raw
fullinertia you wrote in the XML. At compile time it diagonalizes every body's
inertia, storing the principal moments in ``model.body_inertia`` (a 3-vector)
and the orientation of that principal frame (relative to the body frame) as a
quaternion in ``model.body_iquat``. The body-frame tensor is therefore
    I_body = R * diag(principal_moments) * R^T,
where R is the rotation matrix of ``body_iquat``. (Verified to round-trip back
to the original fullinertia to ~1e-13.)
"""

import numpy as np

from ._shared import load_model_from_xml, body_frame_inertia, get_body_name, validate_inertia


def main() -> None:
    """Compile the model and print mass/inertia report."""
    # Compile the full model (resolves <include> + meshdir="assets").
    XML = "src/diogenes_mjlab/diogenes/xmls/scene.xml"
    model = load_model_from_xml(XML)

    print(f"{'id':>2}  {'body':<16} {'mass(kg)':>10}  "
          f"{'principal inertia (kg*m^2)':>34}  valid")
    print("-" * 80)

    total = 0.0
    for i in range(model.nbody):
        name = get_body_name(model, i)
        mass = float(model.body_mass[i])
        I = model.body_inertia[i].copy()            # (Ixx, Iyy, Izz) principal
        total += mass

        if mass <= 1e-6:
            valid = "n/a (massless)"
        else:
            _, valid = validate_inertia(I)

        flag = "  <-- ~zero mass" if (i != 0 and mass <= 1e-6) else ""
        print(f"{i:>2}  {name:<16} {mass:>10.5f}  "
              f"({I[0]:.3e},{I[1]:.3e},{I[2]:.3e})  {valid}{flag}")

    print("-" * 80)
    print(f"    {'TOTAL':<16} {total:>10.5f} kg "
          f"(includes world body id 0, which is always 0)")

    # ---------------------------------------------------------------------------
    # Full 3x3 inertia tensors (body frame), one block per body.
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FULL 3x3 INERTIA TENSORS (body frame, kg*m^2)")
    print("=" * 80)
    for i in range(model.nbody):
        name = get_body_name(model, i)
        mass = float(model.body_mass[i])
        if mass <= 1e-6:
            # Skip the world body and massless anchors (tensor is ~0 / meaningless).
            print(f"\n[{i}] {name}: massless ({mass:.2e} kg) -- tensor omitted")
            continue
        I_full = body_frame_inertia(model, i)
        # Symmetrize tiny numerical asymmetry for clean display.
        I_full = 0.5 * (I_full + I_full.T)
        print(f"\n[{i}] {name}  (mass {mass:.5f} kg)")
        print("    Ixx Ixy Ixz")
        print("    Iyx Iyy Iyz   =")
        print("    Izx Izy Izz")
        for r in range(3):
            print("    " + "  ".join(f"{I_full[r, c]: .6e}" for c in range(3)))
        # Also report the off-diagonal magnitude as a quick "is it nearly diagonal?"
        off = max(abs(I_full[0, 1]), abs(I_full[0, 2]), abs(I_full[1, 2]))
        print(f"    (max |off-diagonal| = {off:.3e})")

    print("\nDone. Compare these against your Onshape mass-properties panel.")


if __name__ == "__main__":
    main()
