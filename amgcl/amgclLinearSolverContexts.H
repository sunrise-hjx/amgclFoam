/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019 OpenCFD Ltd.
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
    Foam::amgclLinearSolverContexts

Description
    A mesh-based collection linear solver contexts for amgcl

SourceFiles
    amgclLinearSolverContexts.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_module_amgclLinearSolverContexts_H
#define Foam_module_amgclLinearSolverContexts_H

#include "fvMesh.H"
#include "MeshObject.H"
#include "HashPtrTable.H"
#include "amgclLinearSolverContext.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class amgclLinearSolverContexts Declaration
\*---------------------------------------------------------------------------*/

class amgclLinearSolverContexts
:
    public MeshObject
    <
        fvMesh,
        TopologicalMeshObject,
        amgclLinearSolverContexts
    >
{
    // Private Data

        //- Linear solver contexts
        mutable HashPtrTable<amgclLinearSolverContext> contexts_;


public:

    //- Runtime information
    TypeName("amgclLinearSolverContext");


    // Constructors

        //- Construct from mesh for given field
        explicit amgclLinearSolverContexts(const fvMesh& mesh)
        :
            MeshObject
            <
                fvMesh,
                Foam::TopologicalMeshObject,
                amgclLinearSolverContexts
            >(mesh),
            contexts_()
        {}


    //- Destructor
    virtual ~amgclLinearSolverContexts() = default;


    // Member Functions

        //- Linear solver contexts (mutable)
        HashPtrTable<amgclLinearSolverContext>& contexts() const
        {
            return contexts_;
        }


        //- Test if named context exists
        bool haveContext(const word& name) const
        {
            return contexts_.found(name);
        }

        //- Get existing or a new context by name
        amgclLinearSolverContext& getContext(const word& name) const
        {
            auto iter = contexts().find(name);

            const bool cached = iter.found();

            if (!iter.found())
            {
                contexts().set(name, new amgclLinearSolverContext{});
                iter = contexts().find(name);
            }

            DebugInfo
                << "cache-" << (cached ? "hit" : "miss")
                << " amgcl linear solver context "
                << name << " mesh: " << mesh().name() << nl;

            return *(*iter);
        }


        //- Remove context by name
        bool removeContext(const word& name) const
        {
            return contexts().erase(name);
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //



