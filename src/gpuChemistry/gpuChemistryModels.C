/*---------------------------------------------------------------------------*\
  gpuChemistryModel 인스턴스화·런타임 등록.
  선택 키: "gpu<chemistryModel<Thermo>>" (basicChemistryModel::New 규약:
  solver + '<' + method(기본 chemistryModel) + '<' + thermoName + ">>")
\*---------------------------------------------------------------------------*/

#include "gpuChemistryModel.H"

#include "addToRunTimeSelectionTable.H"

// janaf 기반 기체 thermo만 등록 (armGpu가 janaf 계수 접근자를 요구)
#include "specie.H"
#include "perfectGas.H"
#include "janafThermo.H"
#include "sensibleEnthalpy.H"
#include "sensibleInternalEnergy.H"
#include "thermo.H"
#include "sutherlandTransport.H"
#include "constTransport.H"

#define makeGpuChemistryModel(nullArg, ThermoPhysics)                          \
                                                                               \
    typedef gpuChemistryModel<ThermoPhysics>                                   \
        gpuChemistryModel##ThermoPhysics;                                      \
                                                                               \
    defineTemplateTypeNameAndDebugWithName                                     \
    (                                                                          \
        gpuChemistryModel##ThermoPhysics,                                      \
        (                                                                      \
            word(gpuChemistryModel##ThermoPhysics::typeName_())                \
          + "<chemistryModel<" + ThermoPhysics::typeName() + ">>"              \
        ).c_str(),                                                             \
        0                                                                      \
    );                                                                         \
                                                                               \
    addToRunTimeSelectionTable                                                 \
    (                                                                          \
        basicChemistryModel,                                                   \
        gpuChemistryModel##ThermoPhysics,                                      \
        thermo                                                                 \
    )

namespace Foam
{
    typedef sutherlandTransport<species::thermo
        <janafThermo<perfectGas<specie>>, sensibleEnthalpy>>
        gpuGasHsSuth;
    typedef constTransport<species::thermo
        <janafThermo<perfectGas<specie>>, sensibleEnthalpy>>
        gpuGasHsConst;
    typedef sutherlandTransport<species::thermo
        <janafThermo<perfectGas<specie>>, sensibleInternalEnergy>>
        gpuGasEsSuth;

    makeGpuChemistryModel(nullArg, gpuGasHsSuth);
    makeGpuChemistryModel(nullArg, gpuGasHsConst);
    makeGpuChemistryModel(nullArg, gpuGasEsSuth);
}

// ************************************************************************* //
