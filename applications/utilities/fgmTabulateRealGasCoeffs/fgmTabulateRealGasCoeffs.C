/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2026 RGP-13
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    fgmTabulateRealGasCoeffs

Description
    Pre-tabulate the composition-dependent real-gas mixture coefficients (the
    O(n^2) Chung/SRK pair-mixing output of SRKchungTakaMixture::calculateRealGas)
    onto the existing FGM manifold and append them to constant/fgmProperties as
    the RG_<name> fields.

    This is the offline half of the FPV/FGM "Tier 2" optimisation: the 13
    coefficients
        bM, coef1, coef2, coef3, cM,                 (SRK EoS)
        sigmaM, epsilonkM, MM, VcM, TcM, omegaM,     (Chung mu/kappa)
        miuiM, kappaiM
    are pure functions of the mole-fraction composition X = X(Z, gZ, c[, chi]),
    so they are tabulated once here using the REAL mixture object (guaranteeing
    bit-consistency with the live path) and then looked up per cell by fgmFluid
    instead of being rebuilt with the O(n^2) pair sum every step.

    The composition at each manifold node is taken from the tabulated species
    mass fractions Y_<sp> already in fgmProperties; species present in the
    thermo but absent from the table are treated as Y = 0 (consistent with the
    solver, which never advances untabulated species off zero).

Usage
    Run in a case whose constant/fgmProperties already carries the species (Y_*)
    tables and whose physicalProperties selects an SRKchungTakaMixture:
        fgmTabulateRealGasCoeffs
    constant/fgmProperties is rewritten in place with the RG_* fields added.

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "fvMesh.H"
#include "Time.H"
#include "fluidMulticomponentThermo.H"
#include "tabulatedRealGasMixture.H"
#include "IOdictionary.H"
#include "OSspecific.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Pre-tabulate the SRK/Chung real-gas mixture coefficients onto the FGM"
        " manifold (RG_* fields appended to constant/fgmProperties)."
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // Build the real-fluid multicomponent thermo exactly as the solver does, so
    // the mixture carries the case's actual species constants and mixing rules.
    Info<< "Constructing fluidMulticomponentThermo ..." << endl;
    autoPtr<fluidMulticomponentThermo> pThermo
    (
        fluidMulticomponentThermo::New(mesh)
    );
    fluidMulticomponentThermo& thermo = pThermo();

    // The mixture must implement the tabulation hook (SRKchungTaka backend).
    const tabulatedRealGasMixture* hook =
        dynamic_cast<const tabulatedRealGasMixture*>(&thermo);
    if (!hook)
    {
        FatalErrorInFunction
            << "The active mixture does not implement tabulatedRealGasMixture."
            << nl << "physicalProperties must select mixture "
            << "SRKchungTakaMixture for Tier-2 coefficient tabulation."
            << exit(FatalError);
    }

    // Full thermo species list (the composition order used by calcMixture).
    const wordList thermoSpecies(thermo.species());
    const label nSp = thermoSpecies.size();

    // ---- read the existing FGM table ----
    IOdictionary fgm
    (
        IOobject
        (
            "fgmProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        )
    );

    const label nZ  = readLabel(fgm.lookup("nZ"));
    const label nGz = readLabel(fgm.lookup("nGz"));
    const label nC  = readLabel(fgm.lookup("nC"));
    label nChi = 1;
    if (fgm.found("nChi"))
    {
        nChi = readLabel(fgm.lookup("nChi"));
    }
    else if (fgm.found("nH"))
    {
        nChi = readLabel(fgm.lookup("nH"));
    }
    const label nTot = nZ*nGz*nC*nChi;

    Info<< "FGM grid: nZ=" << nZ << " nGz=" << nGz << " nC=" << nC
        << " nChi=" << nChi << "  (" << nTot << " nodes)" << endl;

    if (!fgm.found("species"))
    {
        FatalErrorInFunction
            << "fgmProperties has no 'species' entry -- the Y_<sp> tables are "
            << "required to build the composition." << exit(FatalError);
    }
    const wordList tabSpecies(fgm.lookup("species"));

    // Load the tabulated species mass-fraction fields.
    HashTable<List<scalar>> Ytab;
    forAll(tabSpecies, i)
    {
        List<scalar> t(fgm.lookup("Y_" + tabSpecies[i]));
        if (t.size() != nTot)
        {
            FatalErrorInFunction
                << "Y_" << tabSpecies[i] << " size " << t.size()
                << " != nTot " << nTot << exit(FatalError);
        }
        Ytab.insert(tabSpecies[i], t);
    }

    // Map each thermo species to its table (null if untabulated -> Y = 0).
    List<const List<scalar>*> spTab(nSp, static_cast<const List<scalar>*>(0));
    label nMissing = 0;
    forAll(thermoSpecies, j)
    {
        if (Ytab.found(thermoSpecies[j]))
        {
            spTab[j] = &Ytab[thermoSpecies[j]];
        }
        else
        {
            nMissing++;
        }
    }
    Info<< thermoSpecies.size() - nMissing << " of " << nSp
        << " thermo species are tabulated; " << nMissing
        << " treated as Y = 0." << endl;

    // ---- compute the coefficients per manifold node ----
    const wordList& cn = tabulatedRealGasMixture::coeffNames();
    const label nCoeff = cn.size();

    List<List<scalar>> RG(nCoeff);
    forAll(RG, k)
    {
        RG[k].setSize(nTot);
    }

    List<scalar> Yc(nSp);
    List<scalar> coeffs(nCoeff);

    Info<< "Tabulating " << nCoeff << " real-gas coefficients over "
        << nTot << " nodes ..." << endl;

    for (label idx = 0; idx < nTot; idx++)
    {
        forAll(Yc, j)
        {
            Yc[j] = spTab[j] ? (*spTab[j])[idx] : scalar(0);
        }

        hook->realGasCoeffs(Yc, coeffs);

        forAll(RG, k)
        {
            RG[k][idx] = coeffs[k];
        }
    }

    // ---- back up the original table once, then append RG_* and rewrite ----
    // Never overwrite an existing backup, so the pristine pre-RG table survives
    // repeated runs. (The builder's .npz companion is the ultimate source.)
    const fileName bak(fgm.objectPath() + ".preRG");
    if (!isFile(bak))
    {
        Foam::cp(fgm.objectPath(), bak);
        Info<< "Backed up original table to " << bak << endl;
    }

    forAll(cn, k)
    {
        const word key("RG_" + cn[k]);
        fgm.remove(key);          // drop any stale copy from a previous run
        fgm.add(key, RG[k]);
    }

    Info<< "Writing " << nCoeff << " RG_* tables to constant/fgmProperties ..."
        << endl;
    fgm.regIOobject::write();

    Info<< nl << "Done. fgmProperties now carries the Tier-2 RG_* coefficients."
        << nl << "End" << nl << endl;

    return 0;
}


// ************************************************************************* //
