/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2022 OpenCFD Ltd.
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

Class
    Foam::amgclLinearSolverContext

Description
    A struct containing a KSP and Mat for each equation to be solved

SourceFiles
    amgclLinearSolverContext.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_module_amgclLinearSolverContext_H
#define Foam_module_amgclLinearSolverContext_H

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
#include <amgcl/mpi/solver/cg.hpp>

#include "solverPerformance.H"
#include "amgclCacheManager.H"
#include "amgclUtils.H"
#include "globalIndex.H"

// amgcl
// #include "petscvec.h"
// #include "petscmat.h"
// #include "petscksp.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class amgclLinearSolverContext Declaration
\*---------------------------------------------------------------------------*/

class amgclLinearSolverContext
{
    // Private Data

        bool init_;
        bool init_aux_vectors_;
        doubleScalar IniResidual;

    public:
        // Public Members

        //- Using OpenFOAM norm for convergence testing
        bool useFoamTest;
        doubleScalar error;

        // Mat Amat;
        // KSP ksp;
        // typedef amgcl::backend::builtin<double> SBackend;
        // typedef amgcl::backend::builtin<double> PBackend;
        // typedef amgcl::make_solver<
        //     amgcl::amg<
        //         PBackend,
        //         amgcl::coarsening::smoothed_aggregation,
        //         amgcl::relaxation::spai0
        //         >,
        //     amgcl::solver::bicgstab<SBackend>
        //     > Solver;

        std::vector<ptrdiff_t> ptr = std::vector<ptrdiff_t> (1,1);
        std::vector<ptrdiff_t> col = std::vector<ptrdiff_t> (1,1);
        std::vector<double> val = std::vector<double> (1,1.0);
        std::vector<double> rhs = std::vector<double> (1,1.0);
        std::vector<double> AMGx = std::vector<double> (1,1.0);
        std::vector<double> AMGr = std::vector<double> (1,1.0);

        labelList ltg_;

        solverPerformance performance;

        amgclCacheManager caching;

        List<int> lowNonZero;
        int maxLowNonZeroPerRow = 0;

        doubleScalar normFactor;

        int matstage;
        int pcstage;
        int kspstage;


    // Constructors

        //- Default construct
        amgclLinearSolverContext()
        :
            init_(false),
            init_aux_vectors_(false),
            useFoamTest(false),
            IniResidual(1.0),
            error(1.0)
        {}


    //- Destructor
    virtual ~amgclLinearSolverContext(){}

    // Member Functions

        //- Return value of initialized
        bool initialized() const noexcept
        {
            return init_;
        }

        //- Change state of initialized, return previous value
        bool initialized(const bool yes) noexcept
        {
            bool old(init_);
            init_ = yes;
            return old;
        }

        void calIniResidual
        (
            std::vector<ptrdiff_t>& ptr,
            std::vector<ptrdiff_t>& col,
            std::vector<double>& val,
            std::vector<double>& rhs,
            solveScalarField& psi
        )
        {
            doubleScalar up = 0.0;
            for (label i = 0; i < rhs.size(); i++)
            {
                doubleScalar sum = 0.0;
                for (label ptri = ptr[i]; ptri < ptr[i + 1]; ptri++)
                {
                    sum += psi[col[ptri]] * val[ptri];
                }
                up += Foam::mag(rhs[i] - sum);
            }
            IniResidual = up;
        }

        void set_ltg
        (
            label &numCells
        )
        {
            ltg_.setSize(numCells);
            const globalIndex globalnumber(numCells);
            for (label i = 0; i < numCells; i++)
            {
                ltg_[i] = globalnumber.toGlobal(i);
            }
        }

        doubleScalar getIniResidual()
        {
            return IniResidual;
        }

        //- Compute residual (L1) norm
        //- assumes normFactor has been already computed
        double getResidualNormL1(std::vector<double> AMGx_, std::vector<double> rhs_)
        {
            // if (!init_aux_vectors_) return static_cast<double>(-1);

            double rnorm = 0.0;
            for (size_t li = 0; li < rhs_.size(); li++)
            {
                rnorm += (AMGx_[li] - rhs_[li]) > 0 ? (AMGx_[li] - rhs_[li]) : (rhs_[li] - AMGx_[li]);
            }
            
            return rnorm/normFactor;
        }


    // Housekeeping

        //- Non-const access to state of initialized
        bool& initialized() noexcept
        {
            return init_;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
