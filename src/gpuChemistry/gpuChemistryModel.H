/*---------------------------------------------------------------------------*\
Class
    Foam::gpuChemistryModel

Description
    chemistryModel의 셀 루프(강성 ODE 적분)를 chemGPU 배치 커널
    (rgpChemIntegrate — Rosenbrock34, OF와 동일 산술)로 오프로드하는
    화학 모델. constant/chemistryProperties에서

        chemistryType { solver gpu; }

    로 선택한다. CPU 수송이 조립되는 동안 GPU가 화학을 푸는 이기종
    분담(단기 로드맵)의 솔버 쪽 절반.

    v1 제약: 메커니즘은 Burke 2012 H2/O2(rgpH2O2Burke.H) 고정 —
    케이스 종 목록이 9종과 이름으로 정확히 일치해야 하며(순서 무관),
    불일치 시 FatalError. (OF reactions 파싱 → POD export는 후속.)

SourceFiles
    gpuChemistryModel.C
\*---------------------------------------------------------------------------*/

#ifndef gpuChemistryModel_H
#define gpuChemistryModel_H

#include "chemistryModel.H"

#include "gpu/rgpChemTypes.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class gpuChemistryModel Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType>
class gpuChemistryModel
:
    public chemistryModel<ThermoType>
{
    // Private data

        //- 커널 허용오차 (chemistryProperties gpuCoeffs 서브딕셔너리)
        scalar relTol_, absTol_;

        //- retrieve 캐시 허용오차 (gpuCoeffs retrieveTol, 0=off)
        scalar retrieveTol_;

        //- 메커니즘(디바이스 업로드 원본)과 종 인덱스 맵
        mutable rgpChemMech mech_;
        mutable labelList map_;      // mech 종 m -> thermo 종 인덱스
        mutable bool armed_;

        //- grow-only 호스트 gather/scatter 버퍼
        mutable List<double> pB_, TB_, cB_, dtB_;


    // Private Member Functions

        //- 1회 arming: 종 이름 매핑 검증 + 디바이스 업로드
        void armGpu() const;

        //- 배치 적분 + RR 갱신 (균일/셀별 dt 공용)
        template<class DeltaTType>
        scalar solveBatch(const DeltaTType& deltaT);


public:

    //- Runtime type information
    TypeName("gpu");


    // Constructors

        //- Construct from thermo
        gpuChemistryModel(const fluidMulticomponentThermo& thermo);

        //- Disallow default bitwise copy construction
        gpuChemistryModel(const gpuChemistryModel&) = delete;


    //- Destructor
    virtual ~gpuChemistryModel();


    // Member Functions

        //- Solve the reaction system for the given time step
        virtual scalar solve(const scalar deltaT);

        //- Solve the reaction system for the given (per-cell) time step
        virtual scalar solve(const scalarField& deltaT);

        //- 셀 단위 ODE solve (chemistryModel 순수가상) — 호스트 폴백.
        //  배치 경로는 이 함수를 거치지 않는다.
        virtual void solve
        (
            scalar& p,
            scalar& T,
            scalarField& Y,
            const label li,
            scalar& deltaT,
            scalar& subDeltaT
        ) const;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const gpuChemistryModel&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "gpuChemistryModel.C"
#endif

#endif

// ************************************************************************* //
