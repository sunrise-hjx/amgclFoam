/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2022 OpenCFD Ltd.
    Copyright (C) 2019-2020 Simone Bna
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
    Foam::amgclSolver

Group
    grpLduMatrixSolvers

Description
    amgcl solver interface for lduMatrices using a run-time selectable solver
    and preconditioner. amgcl solvers are configured via sub-dictionaries.
    For example,

    \verbatim
    solver amgcl;

    amgcl
    {
        use_amgcl_residual_norm true;

        options
        {
            ksp_type cg;
            ksp_cg_single_reduction true;
        }
    }
    \endverbatim

SourceFiles
    amgclSolver.C

\*---------------------------------------------------------------------------*/

// #ifndef Foam_module_amgclSolver_H
// #define Foam_module_amgclSolver_H

#ifndef amgclSolver_H
#define amgclSolver_H

#include "lduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class amgclSolver Declaration
\*---------------------------------------------------------------------------*/

class amgclSolver:public lduMatrix::solver
{
    // Private Data

        //- Coefficients for the amgcl solvers
        dictionary amgclDict_;

        //- Equation name for solver context
        word eqName_;

        //- Solution name used as prefix
        word prefix_;

    // Private Member Functions

        //- Compute allocation (proc and off-proc)
        //- for the amgcl matrix Amat_
        void buildMat
        (
            std::vector<ptrdiff_t>& ptrb,
            std::vector<ptrdiff_t>& colb,
            std::vector<double>& valb,
            std::vector<double>& rhsb,
            std::vector<double>& AMGxb,
            std::vector<double>& AMGrb,
            const labelList& paraStartNum
        ) const;

        //- Update Amat_ matrix by inserting new values
        void updateMat
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
        ) const;

        //- No copy construct
        amgclSolver(const amgclSolver&) = delete;

        //- No copy assignment
        void operator=(const amgclSolver&) = delete;


public:

    //- Runtime type information
    TypeName("amgclMPI");

    // Constructors

        //- Construct from matrix components and solver controls
        amgclSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~amgclSolver() = default;

    // Member Functions

        //- Solve with given field and rhs (in solveScalar precision). 
        virtual solverPerformance scalarSolve
        (
            solveScalarField& psi,
            const solveScalarField& source,
            const direction cmpt = 0
        ) const;

        //- Solve the matrix with this solver 
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt = 0
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
