/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2022 OpenCFD Ltd.
    Copyright (C) 2019-2020 Simone Bna
    Copyright (C) 2021 Stefano Zampini
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include <vector>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>

// Serial coarsening
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>

// Serial relaxation
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/spai1.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/iluk.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>

// Serial solvers
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/solver/lgmres.hpp>
#include <amgcl/solver/idrs.hpp>

// MPI support
#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/amg.hpp>
#include <amgcl/mpi/coarsening/smoothed_aggregation.hpp>
#include <amgcl/mpi/relaxation/spai0.hpp>
#include <amgcl/mpi/relaxation/spai1.hpp>
#include <amgcl/mpi/solver/cg.hpp>
#include <amgcl/mpi/solver/bicgstab.hpp>
#include <amgcl/mpi/solver/bicgstabl.hpp>
#include <amgcl/mpi/solver/gmres.hpp>

#include <amgcl/profiler.hpp>

#include "fvMesh.H"
#include "fvMatrices.H"
#include "globalIndex.H"
#include "PrecisionAdaptor.H"
#include "cyclicLduInterface.H"
#include "cyclicAMILduInterface.H"
#include "PstreamGlobals.H"
#include "addToRunTimeSelectionTable.H"
#include "syncTools.H"
#include "processorFvPatch.H"

#include "amgclSolver.H"
#include "amgclControls.H"
#include "amgclLinearSolverContexts.H"
#include "amgclUtils.H"

#include <algorithm>  // For std::max_element
#include <cstring>    // For NULL

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(amgclSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<amgclSolver>
        addamgclSolverSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<amgclSolver>
        addamgclSolverAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * Helper Functions * * * * * * * * * * * * * //

namespace
{
    // Base template - only set common parameters
    template<typename Params>
    void setAMGCLParametersBase
    (
        Params& prm,
        const Foam::dictionary& dict,
        const Foam::solveScalar& relTol,
        const Foam::solveScalar& absTol
    )
    {
        using namespace Foam;
        
        // Basic solver parameters (all solvers have these)
        prm.solver.tol = relTol;
        prm.solver.abstol = absTol;
        prm.solver.maxiter = dict.getOrDefault<label>("maxiter", 100);
        
        // AMG coarsening parameters (if available)
        dictionary amgDict = dict.subOrEmptyDict("amg");
        if (amgDict.found("eps_strong"))
        {
            prm.precond.coarsening.aggr.eps_strong = amgDict.get<scalar>("eps_strong");
        }
    }
    
    // Specializations for different solvers with M parameter (GMRES, FGMRES)
    template<typename Backend>
    void setSolverM(amgcl::solver::gmres<Backend>& solver, const Foam::dictionary& dict)
    {
        if (dict.found("M"))
        {
            solver.M = dict.get<Foam::label>("M");
        }
    }
    
    template<typename Backend>
    void setSolverM(amgcl::solver::fgmres<Backend>& solver, const Foam::dictionary& dict)
    {
        if (dict.found("M"))
        {
            solver.M = dict.get<Foam::label>("M");
        }
    }
    
    // For solvers without M parameter - do nothing
    template<typename Solver>
    void setSolverM(Solver&, const Foam::dictionary&) {}
    
    // Specializations for BiCGStab(L) with L parameter
    template<typename Backend>
    void setSolverL(amgcl::solver::bicgstabl<Backend>& solver, const Foam::dictionary& dict)
    {
        if (dict.found("L"))
        {
            solver.L = dict.get<Foam::label>("L");
        }
    }
    
    // For solvers without L parameter - do nothing
    template<typename Solver>
    void setSolverL(Solver&, const Foam::dictionary&) {}
    
    // For BiCGStab with ns_search
    template<typename Backend>
    void setSolverNsSearch(amgcl::solver::bicgstab<Backend>& solver, const Foam::dictionary& dict)
    {
        if (dict.found("ns_search"))
        {
            solver.ns_search = dict.get<bool>("ns_search");
        }
    }
    
    // For solvers without ns_search - do nothing
    template<typename Solver>
    void setSolverNsSearch(Solver&, const Foam::dictionary&) {}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::amgclSolver::amgclSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    amgclDict_(solverControls.subDict("amgcl")),
    eqName_(fieldName),
    prefix_("eqn_" + eqName_ + "_")
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::amgclSolver::scalarSolve
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt
) const
{
    // Check if running in parallel
    const bool parallel = Pstream::parRun();
    
    const label nPro = Pstream::nProcs();
    const label myProNo = Pstream::myProcNo();
    const fvMesh& mesh = dynamicCast<const fvMesh>(matrix_.mesh().thisDb());
    labelList paraStartNum(nPro + 1, 0);
    const amgclControls &controls = amgclControls::New(matrix_.mesh().thisDb().time());
    dictionary amgclDictOptions = amgclDict_.subOrEmptyDict("options");
    const fvMesh& fvm = dynamicCast<const fvMesh>(matrix_.mesh().thisDb());
    const amgclLinearSolverContexts &contexts = amgclLinearSolverContexts::New(fvm);
    amgclLinearSolverContext& ctx = contexts.getContext(eqName_);
    const bool firsttimein = !ctx.initialized();
    dictionary amgclDictCaching = amgclDict_.subOrEmptyDict("caching");

    std::vector<ptrdiff_t> &ptrs = ctx.ptr;
    std::vector<ptrdiff_t> &cols = ctx.col;
    std::vector<double> &vals = ctx.val;
    std::vector<double> &rhss = ctx.rhs;
    std::vector<double> &AMGxs = ctx.AMGx;
    std::vector<double> &AMGrs = ctx.AMGr;
    label An = source.size();

    const globalIndex GN(An);

    paraStartNum[0] = 0;
    for (int i = 0; i < nPro; i++)
    {
        paraStartNum[i + 1] = GN.localSize(i) + paraStartNum[i];
    }
    const label AMGlocalStart = paraStartNum[myProNo];
    const lduInterfacePtrsList interfaces(matrix_.mesh().interfaces());

    labelField proNei(mesh.nFaces(),0);
    forAll(proNei,faceI)
    {
        proNei[faceI] = Pstream::myProcNo();
    }
    syncTools::swapFaceList(mesh, proNei);
    labelField proNeiPatch(mesh.boundary().size(), Pstream::myProcNo());
    forAll(proNeiPatch,patchI)
    {
        label neiProNo = 0;
        forAll(mesh.boundary()[patchI], facei)
        {
            const label& faceI = mesh.boundaryMesh()[patchI].start() + facei;
            neiProNo = proNei[faceI];
            break;
        }
        proNeiPatch[patchI] = neiProNo;
    }
    proNei.clear();

    solverPerformance solverPerf
    (
        "amgcl",
        fieldName_
    );
    ctx.performance = solverPerf;
    
    if (firsttimein)
    {
        ctx.caching.init(amgclDictCaching);
        ctx.caching.eventBegin();
        Info << "Initializing amgcl Linear Solver " << eqName_ << nl;
        ctx.initialized(true);
        amgcl::profiler<> prof("VPM");
        buildMat(ptrs, cols, vals, rhss, AMGxs, AMGrs, paraStartNum,proNeiPatch);
        ctx.set_ltg(An);

        forAll(psi, pi)
        {
            AMGxs[pi] = psi[pi];
        }
    }

    typedef amgcl::backend::builtin<double> DBackend;
    typedef amgcl::backend::builtin<double> FBackend;

    if(ctx.caching.needsMatrixUpdate())
    {
        Info << "Update Matrix for: " << eqName_ << "; " << nl;
        labelList &ltg = ctx.ltg_;
        updateMat(ptrs, cols, vals, rhss, AMGxs, fvm, source, ltg, paraStartNum,proNeiPatch);
    }
    forAll(psi,ri)
    {
        rhss[ri] = source[ri];
    }

    // Calculate initial residual using OpenFOAM standard method
    solveScalarField Apsi(psi.size());
    solveScalarField temp(psi.size());
    
    matrix_.Amul(Apsi, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    temp = source - Apsi;
    
    solveScalar normFactor = this->normFactor(psi, source, Apsi, temp);
    ctx.normFactor = normFactor;
    
    solveScalar initialResidual = gSumMag(temp, matrix_.mesh().comm()) / normFactor;
    
    size_t AMGiters;
    doubleScalar &AMGerror = ctx.error;
    ctx.performance.initialResidual() = initialResidual;

    // Get solver type from dictionary
    word solverType = amgclDictOptions.getOrDefault<word>("solver", "bicgstab");
    word coarseningType = amgclDictOptions.getOrDefault<word>("coarsening", "smoothed_aggregation");
    word relaxationType = amgclDictOptions.getOrDefault<word>("relaxation", "spai0");
    
    // Print solver configuration
    if (firsttimein)
    {
        Info<< "AMGCL Solver Configuration for " << eqName_ << ":" << nl
            << "  Solver: " << solverType << nl
            << "  Coarsening: " << coarseningType << nl
            << "  Relaxation: " << relaxationType << nl
            << "  Parallel: " << (parallel ? "Yes" : "No") << nl;
    }

    if (parallel)
    {
        // Parallel solver using MPI
        amgcl::mpi::init mpi();
        amgcl::mpi::communicator AMGCLworld(MPI_COMM_WORLD);
        
        auto Acl = std::make_shared<amgcl::mpi::distributed_matrix<DBackend>>(
            AMGCLworld, std::tie(An, ptrs, cols, vals)
        );
        
        // Select solver type for parallel
        if (solverType == "cg")
        {
            typedef amgcl::mpi::make_solver<
                amgcl::mpi::amg<FBackend, 
                    amgcl::mpi::coarsening::smoothed_aggregation<FBackend>,
                    amgcl::mpi::relaxation::spai0<FBackend>>,
                amgcl::mpi::solver::cg<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            Solver AMGsolve(AMGCLworld, Acl, prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(*Acl, rhss, AMGxs);
        }
        else if (solverType == "gmres")
        {
            typedef amgcl::mpi::make_solver<
                amgcl::mpi::amg<FBackend,
                    amgcl::mpi::coarsening::smoothed_aggregation<FBackend>,
                    amgcl::mpi::relaxation::spai0<FBackend>>,
                amgcl::mpi::solver::gmres<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            prm.solver.M = amgclDictOptions.getOrDefault<label>("M", 30);
            Solver AMGsolve(AMGCLworld, Acl, prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(*Acl, rhss, AMGxs);
        }
        else if (solverType == "bicgstabl")
        {
            typedef amgcl::mpi::make_solver<
                amgcl::mpi::amg<FBackend,
                    amgcl::mpi::coarsening::smoothed_aggregation<FBackend>,
                    amgcl::mpi::relaxation::spai0<FBackend>>,
                amgcl::mpi::solver::bicgstabl<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            prm.solver.L = amgclDictOptions.getOrDefault<label>("L", 2);
            Solver AMGsolve(AMGCLworld, Acl, prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(*Acl, rhss, AMGxs);
        }
        else  // Default: bicgstab
        {
            typedef amgcl::mpi::make_solver<
                amgcl::mpi::amg<FBackend,
                    amgcl::mpi::coarsening::smoothed_aggregation<FBackend>,
                    amgcl::mpi::relaxation::spai0<FBackend>>,
                amgcl::mpi::solver::bicgstab<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            if (amgclDictOptions.found("ns_search"))
            {
                prm.solver.ns_search = amgclDictOptions.get<bool>("ns_search");
            }
            Solver AMGsolve(AMGCLworld, Acl, prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(*Acl, rhss, AMGxs);
        }
    }
    else
    {
        // Serial solver
        // Select solver and relaxation type for serial
        if (solverType == "cg" && relaxationType == "spai0")
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
                amgcl::solver::cg<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
        else if (solverType == "cg" && relaxationType == "spai1")
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai1>,
                amgcl::solver::cg<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
        else if (solverType == "cg" && relaxationType == "ilu0")
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::ilu0>,
                amgcl::solver::cg<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
        else if (solverType == "gmres" && relaxationType == "spai0")
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
                amgcl::solver::gmres<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            prm.solver.M = amgclDictOptions.getOrDefault<label>("M", 30);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
        else if (solverType == "gmres" && relaxationType == "ilu0")
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::ilu0>,
                amgcl::solver::gmres<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            prm.solver.M = amgclDictOptions.getOrDefault<label>("M", 30);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
        else if (solverType == "bicgstabl")
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
                amgcl::solver::bicgstabl<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            prm.solver.L = amgclDictOptions.getOrDefault<label>("L", 2);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
        else if (solverType == "fgmres")
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
                amgcl::solver::fgmres<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            prm.solver.M = amgclDictOptions.getOrDefault<label>("M", 30);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
        else if (solverType == "idrs")
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
                amgcl::solver::idrs<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
        else if (solverType == "bicgstab" && relaxationType == "spai1")
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai1>,
                amgcl::solver::bicgstab<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
        else  // Default: bicgstab with spai0
        {
            typedef amgcl::make_solver<
                amgcl::amg<DBackend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
                amgcl::solver::bicgstab<DBackend>> Solver;
            
            Solver::params prm;
            setAMGCLParametersBase(prm, amgclDictOptions, relTol_, tolerance_);
            Solver AMGsolve(std::tie(An, ptrs, cols, vals), prm);
            std::tie(AMGiters, AMGerror) = AMGsolve(rhss, AMGxs);
        }
    }

    forAll(psi, pi)
    {
        psi[pi] = AMGxs[pi];
    }

    // Calculate final residual using OpenFOAM standard method
    matrix_.Amul(Apsi, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    temp = source - Apsi;
    solveScalar finalResidual = gSumMag(temp, matrix_.mesh().comm()) / normFactor;
    
    ctx.performance.finalResidual() = finalResidual;
    ctx.caching.eventEnd();
    ctx.performance.nIterations() = AMGiters;
    
    if (lduMatrix::debug >= 2)
    {
        Info<< "amgclSolver: " << fieldName_
            << ", Initial residual = " << initialResidual
            << ", Final residual = " << finalResidual
            << ", No Iterations " << AMGiters
            << endl;
    }
    
    return ctx.performance;
}

Foam::solverPerformance Foam::amgclSolver::solve
(
    scalarField& psi_s,
    const scalarField& source,
    const direction cmpt
) const
{
    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);
    return scalarSolve
    (
        tpsi.ref(),
        ConstPrecisionAdaptor<solveScalar, scalar>(source)(),
        cmpt
    );
}


void Foam::amgclSolver::buildMat
(
    std::vector<ptrdiff_t>& ptrb,
    std::vector<ptrdiff_t>& colb,
    std::vector<double>& valb,
    std::vector<double>& rhsb,
    std::vector<double>& AMGxb,
    std::vector<double>& AMGrb,
    const labelList& paraStartNum,
    labelField& proNeiPatch
) const
{
    
    const fvMesh& mesh = dynamicCast<const fvMesh>(matrix_.mesh().thisDb());
    const lduAddressing& lduAddr = matrix_.mesh().lduAddr();
    const lduInterfacePtrsList interfaces(matrix_.mesh().interfaces());
    const label AMGlocalStart = paraStartNum[Pstream::myProcNo()];
    const fvBoundaryMesh bMesh(mesh);
    label n = matrix_.diag().size();
    const globalIndex global(n);
    label num_nonzero = 0;
    List<std::vector<label>> A(n);
    label a_row = 1;
    label a_col = 1;
    bool repeat = 0;

    for(int i=0; i<n; i++)
    {
        a_row = i;
        a_col = AMGlocalStart + i;
        repeat = 0;
        for (int aii = 0; aii < A[a_row].size(); aii++)
        {
            if (A[a_row][aii]==a_col)
            {
                repeat = 1;
            }
        }
        if (!repeat) A[a_row].push_back(a_col);
    }
    for(label faceI=0; faceI<matrix_.lduAddr().lowerAddr().size(); faceI++)
    {
        label l = matrix_.lduAddr().lowerAddr()[faceI];
        label u = matrix_.lduAddr().upperAddr()[faceI];

        a_row = l;
        a_col = AMGlocalStart + u;
        repeat = 0;
        for (int aii = 0; aii < A[a_row].size(); aii++)
        {
            if (A[a_row][aii]==a_col)
            {
                repeat = 1;
            }
        }
        if (!repeat) A[a_row].push_back(a_col);

        a_row = u;
        a_col = AMGlocalStart + l;
        repeat = 0;
        for (int aii = 0; aii < A[a_row].size(); aii++)
        {
            if (A[a_row][aii]==a_col)
            {
                repeat = 1;
            }
        }
        if (!repeat) A[a_row].push_back(a_col);
    }
    labelList globalCells
    (
        identity
        (
            global.localSize(),
            global.localStart()
        )
    );
    // Connections to neighbouring processors
    {
        // Initialise transfer of global cells
        forAll(interfaces, patchi)
        {
            if (interfaces.set(patchi))
            {
                interfaces[patchi].initInternalFieldTransfer
                (
                    Pstream::commsTypes::blocking,
                    globalCells
                );
            }
        }
        forAll(interfaces, patchi)
        {
            if (interfaces.set(patchi))
            {
                const polyPatch &patch = mesh.boundaryMesh()[patchi];
                //patch.type()
                //processorFvPatch proPatch(patch, bMesh);
                label nbrPro = proNeiPatch[patchi];
                label nbrLocalStart = global.localStart(nbrPro);
                const labelUList& faceCells = lduAddr.patchAddr(patchi);

                labelField nbrCells
                (
                    interfaces[patchi].internalFieldTransfer
                    (
                        Pstream::commsTypes::blocking,
                        globalCells
                    )
                );

                const label off = global.localStart();
                forAll(faceCells, i)
                {
                    a_row = faceCells[i];
                    a_col = paraStartNum[nbrPro] + (nbrCells[i] - nbrLocalStart);
                    repeat = 0;
                    for (int aii = 0; aii < A[a_row].size(); aii++)
                    {
                        if (A[a_row][aii]==a_col)
                        {
                            repeat = 1;
                        }
                    }
                    if (!repeat) A[a_row].push_back(a_col);
                }
            }
        }
    }
    forAll(A,ai)
    {
        num_nonzero += A[ai].size();
    }
    ptrb.resize(n + 1,0);
    colb.resize(num_nonzero,1.0);
    valb.resize(num_nonzero,1.0);
    rhsb.resize(n,0.0);
    AMGxb.resize(n,0.0);
    AMGrb.resize(n,0.0);
    ptrb[0] = 0;
    label numcal = 0;
    forAll(A, i)
    {
        label pI[A[i].size()];
        for (label k = 0; k < A[i].size(); k++)
        {
            pI[k] = A[i][k];
        }
        for (label pi = 0; pi < A[i].size() - 1; pi++)
        {
            for (label pj = 0; pj < A[i].size() - pi - 1; pj++)
            {
                if (pI[pj] > pI[pj + 1])
                {
                    label temp = pI[pj + 1];
                    pI[pj + 1] = pI[pj];
                    pI[pj] = temp;
                }
            }
        }
        for (label m = 0; m < A[i].size(); m++)
        {
            
            numcal++;
            colb[numcal - 1] = pI[m];
            valb[numcal - 1] = 1.0;
        }
        ptrb[i + 1] = numcal;
    }
    forAll(A, cellI)
    {
        A[cellI].clear();
    }
}

void Foam::amgclSolver::updateMat
(
    std::vector<ptrdiff_t>& ptru,
    std::vector<ptrdiff_t>& colu,
    std::vector<double>& valu,
    std::vector<double>& rhsu,
    std::vector<double>& AMGxu,
    const fvMesh& fvmu,
    const solveScalarField& source,
    labelList& ltg,
    const labelList& paraStartNum,
    labelField& proNeiPatch
) const
{

    label AMGrow = 0, AMGcol = 0;
    doubleScalar AMGval = 0.0;
    const lduAddressing &lduAddr = matrix_.mesh().lduAddr();
    const lduInterfacePtrsList interfaces(matrix_.mesh().interfaces());
    const label AMGlocalStart = paraStartNum[Pstream::myProcNo()];
    const labelUList& upp = lduAddr.upperAddr();
    const labelUList& low = lduAddr.lowerAddr();
    const scalarField& diagVal = matrix_.diag();
    const fvBoundaryMesh bMesh(fvmu);

    // Local degrees-of-freedom i.e. number of local rows
    const label nrows_ = lduAddr.size();
    const label nIntFaces_ = upp.size();
    const globalIndex global(nrows_);

    // The diagonal
    for (label celli = 0; celli < diagVal.size(); ++celli)
    {
        doubleScalar val = diagVal[celli];

        AMGrow = celli;
        AMGcol = AMGlocalStart + celli;
        AMGval = val;
        for (label ptri = ptru[AMGrow]; ptri < ptru[AMGrow + 1]; ptri++)
        {
            if (AMGcol == colu[ptri])
            {
                valu[ptri] = AMGval;
                break;
            }
        }
    }

    // upper and lower
    for (label faceI = 0; faceI < matrix_.lduAddr().lowerAddr().size(); faceI++)   
    {
        label l = matrix_.lduAddr().lowerAddr()[faceI];
        label u = matrix_.lduAddr().upperAddr()[faceI];
        
        AMGrow = l; 
        AMGcol = AMGlocalStart + u; 
        AMGval = matrix_.upper()[faceI];
        for (label ptri = ptru[AMGrow]; ptri < ptru[AMGrow + 1]; ptri++)
        {
            if (AMGcol == colu[ptri])
            {
                valu[ptri] = AMGval;
                break;
            }
        }

        AMGrow = u; 
        AMGcol = AMGlocalStart + l;
        AMGval = matrix_.lower()[faceI];
        for (label ptri = ptru[AMGrow]; ptri < ptru[AMGrow + 1]; ptri++)
        {
            if (AMGcol == colu[ptri])
            {
                valu[ptri] = AMGval;
                break;
            }
        }
    }

    labelList globalCells
    (
        identity
        (
            global.localSize(),
            global.localStart()
        )
    );

    // Connections to neighbouring processors
    {
        forAll(interfaces, patchi)
        {
            if (interfaces.set(patchi))
            {
                interfaces[patchi].initInternalFieldTransfer
                (
                    Pstream::commsTypes::blocking,
                    globalCells
                );
            }
        }

        if (Pstream::parRun())
        {
            Pstream::waitRequests();
        }

        forAll(interfaces, patchi)
        {
            
            if (interfaces.set(patchi))
            {
                const polyPatch &patch = fvmu.boundaryMesh()[patchi];
                //processorFvPatch proPatch(patch, bMesh);
                label nbrPro = proNeiPatch[patchi];
                label nbrLocalStart = global.localStart(nbrPro);
                const labelUList& faceCells = lduAddr.patchAddr(patchi);
                const scalarField& bCoeffs = interfaceBouCoeffs_[patchi];

                labelField nbrCells
                (
                    interfaces[patchi].internalFieldTransfer
                    (
                        Pstream::commsTypes::blocking, // blocking
                        globalCells
                    )
                );

                if (faceCells.size() != nbrCells.size())
                {
                    FatalErrorInFunction
                        << "Mismatch in interface sizes (AMI?)" << nl
                        << "Have " << faceCells.size() << " != "
                        << nbrCells.size() << nl
                        << exit(FatalError);
                }

                const label off = global.localStart();
                forAll(faceCells, i)
                {
                    doubleScalar bval = -bCoeffs[i];
                    AMGrow = faceCells[i];
                    AMGcol = paraStartNum[nbrPro] + (nbrCells[i] - nbrLocalStart);
                    for (label ptri = ptru[AMGrow]; ptri < ptru[AMGrow + 1]; ptri++)
                    {
                        if (AMGcol == colu[ptri])
                        {
                            valu[ptri] = bval;
                            break;
                        }
                    }
                }
            }
        }
    }
}


// ************************************************************************* //
