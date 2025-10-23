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

\*---------------------------------------------------------------------------*/

#include "Time.H"
#include "OSspecific.H"

#include "amgclControls.H"
// #include "petscsys.h"
// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(amgclControls, 0);
}

int Foam::amgclControls::loaded_ = 0;


// * * * * * * * * * * * * * Static Member Functions * * * * * * * * * * * * //

void Foam::amgclControls::start(const fileName& optionsFile)
{
    int err = 0;

    if (!loaded_)
    {
        bool called = false;
        //err = PetscInitialized(&called);

        if (called == true)
        {
            // Someone else already called it
            loaded_ = -1;

            Info<< "amgcl already initialized - ignoring any options file"
                << nl;
        }
    }

}


void Foam::amgclControls::stop()
{
    if (loaded_ > 0)
    {
        Info<< "Finalizing amgcl" << nl;
        // PetscFinalize();
        loaded_ = 0;
    }
    else if (!loaded_)
    {
        Info<< "amgcl already finalized" << nl;
    }
}


// * * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * //

Foam::amgclControls::amgclControls(const Time& runTime)
:
    MeshObject<Time, TopologicalMeshObject, amgclControls>(runTime),
    IOdictionary
    (
        IOobject
        (
            amgclControls::typeName,
            runTime.system(),
            runTime,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE,
            false // no register
        )
    )
{
    start(runTime.system()/"amgclOptions");
}


const Foam::amgclControls& Foam::amgclControls::New(const Time& runTime)
{
    return MeshObject<Time, TopologicalMeshObject, amgclControls>::New(runTime);
}


// * * * * * * * * * * * * * * * * Destructor * * * * * * * * * * * * * * * //

Foam::amgclControls::~amgclControls()
{
    amgclControls::stop();
}


// ************************************************************************* //
