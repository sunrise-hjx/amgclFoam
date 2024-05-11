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
#include <amgcl/make_block_solver.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/iluk.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/idrs.hpp>
#include <amgcl/solver/preonly.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/amg.hpp>
#include <amgcl/mpi/coarsening/smoothed_aggregation.hpp>
#include <amgcl/mpi/relaxation/spai0.hpp>
#include <amgcl/mpi/solver/bicgstab.hpp>

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
    amgclDict_(solverControls.subDict("amgclMPI")),
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
    amgcl::mpi::init mpi();
    amgcl::mpi::communicator AMGCLworld(MPI_COMM_WORLD);
    
    const label nPro = Pstream::nProcs();
    const label myProNo = Pstream::myProcNo();
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
        buildMat(ptrs, cols, vals, rhss, AMGxs, AMGrs, paraStartNum);
        ctx.set_ltg(An);

        forAll(psi, pi)
        {
            AMGxs[pi] = psi[pi];
        }
    }

    typedef amgcl::backend::builtin<double> DBackend;
    typedef amgcl::backend::builtin<double> FBackend;
    typedef amgcl::mpi::make_solver<
        amgcl::mpi::amg<
            FBackend,
            amgcl::mpi::coarsening::smoothed_aggregation<FBackend>,
            amgcl::mpi::relaxation::spai0<FBackend>
            >,
        amgcl::mpi::solver::bicgstab<DBackend>
        > Solver;

    labelList &ltg = ctx.ltg_;
    Solver::params prm;
    prm.solver.tol = relTol_;
    prm.solver.abstol = tolerance_;
    // prm.solver.check_after = amgclDictOptions.getOrDefault<bool>("check_after", false);
    prm.solver.ns_search = amgclDictOptions.getOrDefault<bool>("ns_search", true);
    prm.solver.maxiter = amgclDictOptions.getOrDefault<label>("maxiter", 100);

    if(ctx.caching.needsMatrixUpdate())
    {
        Info << "Update Matrix for: " << eqName_ << "; " << nl;
        updateMat(ptrs, cols, vals, rhss, AMGxs, fvm, source, ltg, paraStartNum);
    }
    forAll(psi,ri)
    {
        rhss[ri] = source[ri];
    }
    auto Acl = std::make_shared<amgcl::mpi::distributed_matrix<DBackend>>(AMGCLworld, std::tie(An, ptrs, cols, vals));
    Solver AMGsolve(AMGCLworld, Acl, prm);
    size_t AMGiters;
    doubleScalar &AMGerror = ctx.error;

    // amgcl::backend::residual(rhss, *Acl, AMGxs, AMGrs);
    // ctx.performance.initialResidual() = std::sqrt(amgcl::backend::inner_product(AMGrs, AMGrs));
    ctx.performance.initialResidual() = 1.0;

    std::tie(AMGiters, AMGerror) = AMGsolve(*Acl, rhss, AMGxs);
    forAll(psi, pi)
    {
        psi[pi] = AMGxs[pi];
    }

    // amgcl::backend::residual(rhss, *Acl, AMGxs, AMGrs);
    // ctx.performance.finalResidual() = std::sqrt(amgcl::backend::inner_product(AMGrs, AMGrs));
    ctx.performance.finalResidual() = AMGerror;

    ctx.caching.eventEnd();
    ctx.performance.nIterations() = AMGiters;
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
    const labelList& paraStartNum
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
                    Pstream::commsTypes::nonblocking,
                    globalCells
                );
            }
        }
        forAll(interfaces, patchi)
        {
            if (interfaces.set(patchi))
            {
                const polyPatch &patch = mesh.boundaryMesh()[patchi];
                processorFvPatch proPatch(patch, bMesh);
                label nbrPro = proPatch.neighbProcNo();
                label nbrLocalStart = global.localStart(nbrPro);
                const labelUList& faceCells = lduAddr.patchAddr(patchi);

                labelField nbrCells
                (
                    interfaces[patchi].internalFieldTransfer
                    (
                        Pstream::commsTypes::nonblocking,
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
    const labelList& paraStartNum
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
                    Pstream::commsTypes::nonblocking,
                    globalCells
                );
            }
        }

        // if (Pstream::parRun())
        // {
        //     Pstream::waitRequests();
        // }

        forAll(interfaces, patchi)
        {
            
            if (interfaces.set(patchi))
            {
                const polyPatch &patch = fvmu.boundaryMesh()[patchi];
                processorFvPatch proPatch(patch, bMesh);
                label nbrPro = proPatch.neighbProcNo();
                label nbrLocalStart = global.localStart(nbrPro);
                const labelUList& faceCells = lduAddr.patchAddr(patchi);
                const scalarField& bCoeffs = interfaceBouCoeffs_[patchi];

                labelField nbrCells
                (
                    interfaces[patchi].internalFieldTransfer
                    (
                        Pstream::commsTypes::nonblocking,
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
