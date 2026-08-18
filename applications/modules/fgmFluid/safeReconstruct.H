/*---------------------------------------------------------------------------*\
Description
    Hybrid, degenerate-cell-safe drop-in replacement for
    fvc::reconstruct(surface scalar field) -> volVectorField.

    fvc::reconstruct builds, per cell, D = fvc::surfaceSum(SfHat*Sf) (SfHat=Sf/
    |Sf|) and returns inv(D) & fvc::surfaceSum(SfHat*ssf) (fvcReconstruct.C). D
    is singular whenever a cell's face-normal directions do not span 3-D
    space. On a static/conformal mesh this never happens; on a dynamic AMR
    mesh (refiner topoChanger) a transient hanging-node/degenerate polyhedral
    cell CAN produce a near-zero det(D) for one step, and Tensor::inv()
    (TensorI.H) divides by that determinant with no floor -- with FOAM_SIGFPE
    (on by default) this raises SIGFPE and aborts (observed: pepFull+AMR
    crash at t~450-800, `momentumPredictor()`/`correctPressurePEP()`'s
    faceGradP path, 2026-07-23).

    A pure Tikhonov floor (D -> D+reg*I, first attempt) stops the SIGFPE but
    is NOT enough: a marginally-singular D still inverts to a huge (if
    finite) reconstructed vector, which the LTS Courant controller reacts to
    by collapsing dt (observed min flow time scale oscillating 1e-12..1e-9,
    i.e. the symptom just moved from a crash to a stealthy dt collapse).

    Fix (v2): a scale-invariant condition indicator
        cond = det(D) / (tr(D)/3)^3   in [0,1]  (AM-GM: =1 iff D isotropic,
                                                  ->0 as D degenerates)
    both sides have units of (area)^3 so cond is dimensionless and threshold-
    comparable across cells/times regardless of local cell size or AMR level.
    Cells with cond < condTol (default 1e-3) are DEGENERATE for this
    face-based reconstruction; for those cells only, fall back to the
    caller-supplied cell-based (Green-Gauss) gradient `fallback`, which needs
    no inversion and is bounded by construction. This is the same "restrict
    to the cell-based scheme exactly where the face-based one is unusable"
    idea faceGradP itself is built on, just applied locally at a handful of
    bad cells instead of globally on/off.

    LOCAL, RGP-13-owned fix (no core OpenFOAM file touched) -- kept as a
    header so both momentumPredictor.C (faceGradP, Korteweg) and
    pressureCorrector.C (faceGradP cell-velocity update) share one
    implementation.
\*---------------------------------------------------------------------------*/

#ifndef safeReconstruct_H
#define safeReconstruct_H

#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvcSurfaceIntegrate.H"
#include "extrapolatedCalculatedFvPatchFields.H"

namespace Foam
{

inline tmp<volVectorField> safeReconstruct
(
    const surfaceScalarField& ssf,
    const volVectorField& fallback,
    const scalar condTol = 1e-3
)
{
    const fvMesh& mesh = ssf.mesh();

    const surfaceVectorField SfHat(mesh.Sf()/mesh.magSf());

    // Raw internal (per-cell) fields only -- avoids constructing a
    // GeometricField from fvc::surfaceSum's DimensionedField (internal-only)
    // return type, which has no such 1-arg constructor.
    const tensorField D(fvc::surfaceSum(SfHat*mesh.Sf())().primitiveField());
    const vectorField rhs(fvc::surfaceSum(SfHat*ssf)().primitiveField());
    const vectorField& fb = fallback.primitiveField();

    tmp<volVectorField> trecon
    (
        volVectorField::New
        (
            "safeReconstruct(" + ssf.name() + ')',
            mesh,
            dimensioned<vector>("0", ssf.dimensions()/dimArea, Zero),
            extrapolatedCalculatedFvPatchField<vector>::typeName
        )
    );

    vectorField& Uif = trecon.ref().primitiveFieldRef();

    label nDegenerate = 0;
    forAll(Uif, celli)
    {
        const scalar trD3 = tr(D[celli])/3.0;
        const scalar cond =
            (mag(trD3) > SMALL)
          ? det(D[celli])/pow3(trD3)
          : 0;

        if (cond < condTol)
        {
            Uif[celli] = fb[celli];
            nDegenerate++;
        }
        else
        {
            Uif[celli] = inv(D[celli]) & rhs[celli];
        }
    }

    if (nDegenerate > 0)
    {
        Info<< "safeReconstruct(" << ssf.name() << "): " << nDegenerate
            << " degenerate cell(s) fell back to Green-Gauss grad" << endl;
    }

    trecon.ref().correctBoundaryConditions();

    return trecon;
}

} // End namespace Foam

#endif

// ************************************************************************* //
