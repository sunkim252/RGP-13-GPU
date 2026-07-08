/*---------------------------------------------------------------------------*\
  gpuChemistryModel 인스턴스화·런타임 등록.
  선택 키: "gpu<chemistryModel<Thermo>>" (basicChemistryModel::New 규약:
  solver + '<' + method(기본 chemistryModel) + '<' + thermoName + ">>")
\*---------------------------------------------------------------------------*/

#include "gpuChemistryModel.H"

#include "forGases.H"
#include "addToRunTimeSelectionTable.H"

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
    forCoeffGases(makeGpuChemistryModel, nullArg);
}

// ************************************************************************* //
