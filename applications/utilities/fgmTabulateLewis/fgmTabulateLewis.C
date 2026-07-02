/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2026 RGP-13
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    fgmTabulateLewis

Description
    Tier-4 differential diffusion: tabulate the control-variable Lewis numbers
    Le_Z(Z,gZ,c[,chi/dh]) and Le_C onto the FGM manifold and append them to
    constant/fgmProperties, using the case's OWN real-fluid mixture transport
    (SRK + Chung + Takahashi) evaluated at each node's tabulated (T, Y) and
    the given reference pressure:

        Le_Z = Pr = mu*Cp/kappa      (Z is a conserved scalar diffusing like
                                      heat: rho*D_Z = mu/Le_Z = kappa/Cp)
        Le_C = nu/D_C,  D_C = PV-mass-weighted mixture-averaged Dimix

    Evaluating with the solver's own chungTransport (not an external Cantera
    build) makes the tabulated Le consistent by construction: the solver forms
    rho*D = mu/Le with the SAME mu, so D_Z reduces exactly to the solver's
    kappa/Cp. (The Cantera high-pressure-Chung dense-liquid viscosity deviates
    by up to ~4x from the NIST-validated chungTransport below ~120 K, which
    would otherwise corrupt the cryogenic-core diffusivities.)

    Nodes whose tabulated T is below a floor (default 60 K) or whose transport
    evaluation degenerates are filled with the manifold median; all values are
    clamped to [0.1, 10].

Usage
    Run in a case whose constant/fgmProperties carries the T and Y_* tables and
    whose physicalProperties selects an SRKchungTakaMixture:
        fgmTabulateLewis -pressure 5.25e6 [-pvSpecies '(CO2 CO H2O H2)']
    constant/fgmProperties is rewritten in place with Le_Z/Le_C appended
    (original backed up once to fgmProperties.preLe).

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "fvMesh.H"
#include "Time.H"
#include "fluidMulticomponentThermo.H"
#include "tabulatedRealGasMixture.H"
#include "IOdictionary.H"
#include "OSspecific.H"
#include "SortableList.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scalar median(const List<scalar>& v)
{
    if (v.empty())
    {
        return 1;
    }
    SortableList<scalar> s(v);
    return s[s.size()/2];
}


int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Tabulate the Tier-4 differential-diffusion Lewis numbers Le_Z/Le_C"
        " onto the FGM manifold with the case's own real-fluid transport."
    );
    argList::addOption
    (
        "pressure", "Pa",
        "reference pressure for the transport evaluation (table P_ref)"
    );
    argList::addOption
    (
        "pvSpecies", "(CO2 CO H2O H2)",
        "progress-variable species (default: CO2 CO H2O H2)"
    );
    argList::addOption
    (
        "Tfloor", "K",
        "median-fill nodes below this tabulated T (default 60)"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    if (!args.optionFound("pressure"))
    {
        FatalErrorInFunction
            << "-pressure <Pa> is required (the table's P_ref)."
            << exit(FatalError);
    }
    const scalar pRef = args.optionRead<scalar>("pressure");
    const scalar Tfloor = args.optionLookupOrDefault<scalar>("Tfloor", 60.0);

    wordList pvNames({"CO2", "CO", "H2O", "H2"});
    if (args.optionFound("pvSpecies"))
    {
        pvNames = args.optionRead<wordList>("pvSpecies");
    }

    // Build the real-fluid multicomponent thermo exactly as the solver does.
    Info<< "Constructing fluidMulticomponentThermo ..." << endl;
    autoPtr<fluidMulticomponentThermo> pThermo
    (
        fluidMulticomponentThermo::New(mesh)
    );
    fluidMulticomponentThermo& thermo = pThermo();

    const tabulatedRealGasMixture* hook =
        dynamic_cast<const tabulatedRealGasMixture*>(&thermo);
    if (!hook)
    {
        FatalErrorInFunction
            << "The active mixture does not implement tabulatedRealGasMixture."
            << nl << "physicalProperties must select mixture "
            << "SRKchungTakaMixture." << exit(FatalError);
    }

    const wordList thermoSpecies(thermo.species());
    const label nSp = thermoSpecies.size();

    // PV species -> thermo indices
    labelList pvIds;
    forAll(pvNames, k)
    {
        forAll(thermoSpecies, j)
        {
            if (thermoSpecies[j] == pvNames[k])
            {
                pvIds.append(j);
                break;
            }
        }
    }
    Info<< "PV species for D_C: " << pvNames << " -> " << pvIds.size()
        << " resolved in thermo" << endl;
    if (pvIds.empty())
    {
        FatalErrorInFunction
            << "none of the PV species " << pvNames
            << " exist in the thermo." << exit(FatalError);
    }

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
        << " n4=" << nChi << "  (" << nTot << " nodes), pRef = "
        << pRef << " Pa" << endl;

    if (fgm.found("Le_Z"))
    {
        FatalErrorInFunction
            << "fgmProperties already carries Le_Z -- refusing to "
            << "double-apply (restore the .preLe backup first)."
            << exit(FatalError);
    }
    if (!fgm.found("T"))
    {
        FatalErrorInFunction
            << "fgmProperties has no tabulated T -- required to evaluate the "
            << "transport at each node." << exit(FatalError);
    }
    const List<scalar> Ttab(fgm.lookup("T"));
    if (Ttab.size() != nTot)
    {
        FatalErrorInFunction
            << "T size " << Ttab.size() << " != nTot " << nTot
            << exit(FatalError);
    }

    const wordList tabSpecies(fgm.lookup("species"));
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
    List<const List<scalar>*> spTab(nSp, static_cast<const List<scalar>*>(0));
    forAll(thermoSpecies, j)
    {
        if (Ytab.found(thermoSpecies[j]))
        {
            spTab[j] = &Ytab[thermoSpecies[j]];
        }
    }

    // ---- evaluate per node ----
    List<scalar> LeZ(nTot, scalar(-1));   // -1 = pending median fill
    List<scalar> LeC(nTot, scalar(-1));
    List<scalar> Yc(nSp);
    DynamicList<scalar> goodZ(nTot), goodC(nTot);
    label nCold = 0, nFail = 0;

    Info<< "Evaluating Le_Z/Le_C over " << nTot << " nodes with the live "
        << "SRK+Chung+Takahashi transport ..." << endl;

    for (label idx = 0; idx < nTot; idx++)
    {
        const scalar Tn = Ttab[idx];
        if (Tn < Tfloor)
        {
            nCold++;
            continue;
        }

        forAll(Yc, j)
        {
            Yc[j] = spTab[j] ? (*spTab[j])[idx] : scalar(0);
        }

        scalar lz, lc;
        if (hook->lewisNumbers(Yc, pRef, Tn, pvIds, lz, lc))
        {
            lz = max(scalar(0.1), min(scalar(10), lz));
            lc = max(scalar(0.1), min(scalar(10), lc));
            LeZ[idx] = lz;
            LeC[idx] = lc;
            goodZ.append(lz);
            goodC.append(lc);
        }
        else
        {
            nFail++;
        }

        if (idx > 0 && (idx % 50000) == 0)
        {
            Info<< "    " << idx << "/" << nTot << " nodes ("
                << runTime.elapsedClockTime() << " s)" << endl;
        }
    }

    const scalar medZ = median(goodZ);
    const scalar medC = median(goodC);
    label nFillZ = 0;
    forAll(LeZ, idx)
    {
        if (LeZ[idx] < 0) { LeZ[idx] = medZ; LeC[idx] = medC; nFillZ++; }
    }

    scalar loZ = GREAT, hiZ = -GREAT, loC = GREAT, hiC = -GREAT;
    forAll(LeZ, idx)
    {
        loZ = min(loZ, LeZ[idx]); hiZ = max(hiZ, LeZ[idx]);
        loC = min(loC, LeC[idx]); hiC = max(hiC, LeC[idx]);
    }
    Info<< nl << "Le_Z: median " << medZ << "  range [" << loZ << ", "
        << hiZ << "]" << nl
        << "Le_C: median " << medC << "  range [" << loC << ", " << hiC << "]"
        << nl << "median-filled " << nFillZ << " nodes ("
        << nCold << " below Tfloor=" << Tfloor << " K, "
        << nFail << " degenerate)" << endl;

    // ---- back up once, append, rewrite ----
    const fileName bak(fgm.objectPath() + ".preLe");
    if (!isFile(bak))
    {
        Foam::cp(fgm.objectPath(), bak);
        Info<< "Backed up original table to " << bak << endl;
    }

    fgm.remove("Le_Z");
    fgm.remove("Le_C");
    fgm.add("Le_Z", LeZ);
    fgm.add("Le_C", LeC);

    Info<< "Writing Le_Z/Le_C to constant/fgmProperties ..." << endl;
    fgm.regIOobject::write();

    Info<< nl << "Done. fgmProperties now carries the Tier-4 Le_Z/Le_C "
        << "(solver-consistent transport)." << nl << "End" << nl << endl;

    return 0;
}


// ************************************************************************* //
