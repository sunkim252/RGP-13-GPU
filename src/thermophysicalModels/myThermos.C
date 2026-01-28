#include "psiThermo.H"
#include "pureMixture.H"
// #include "multicomponentMixture.H"
#include "SRKGas.H"
#include "janafThermo.H"
#include "makeFluidThermo.H"
#include "sensibleEnthalpy.H"
#include "specie.H"
#include "sutherlandTransport.H"
#include "takahashiTransport.H"
#include "thermo.H"

namespace Foam {
// Typedef for the SRK based thermo physics
// Type: Transport<species::thermo<Thermo<Equation<Specie>>, Energy>>
typedef sutherlandTransport<
    species::thermo<janafThermo<SRKGas<specie>>, sensibleEnthalpy>>
    SRKspecie;

typedef takahashiTransport<
    species::thermo<janafThermo<SRKGas<specie>>, sensibleEnthalpy>>
    SRKspecieTakahashi;

// Instantiate fluid thermo packages
makeFluidThermo(psiThermo, pureMixture, SRKspecie);
makeFluidThermo(psiThermo, pureMixture, SRKspecieTakahashi);

/*
makeFluidThermo
(
    psiThermo,
    multicomponentMixture,
    SRKspecie
);
*/

} // namespace Foam
