/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2020 OpenCFD Ltd.
    Copyright (C) 2019 Simone Bna
    Copyright (C) 2020 Stefano Zampini
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

#include "lduMatrix.H"
#include "error.H"
#include "amgclUtils.H"
#include "amgclLinearSolverContext.H"

// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //


void Foam::amgclUtils::setFlag
(
    const word& key,
    const word& val,
    const bool verbose
)
{
    if (verbose)
    {
        Info<< key << ' ' << val << nl;
    }

    // PetscOptionsSetValue(NULL, key.c_str(), val.c_str());
}


void Foam::amgclUtils::setFlags
(
    const word& prefix,
    const dictionary& dict,
    const bool verbose
)
{
    for (const entry& e : dict)
    {
        const word key = '-' + prefix + e.keyword();
        const word val = e.get<word>();

        if (verbose)
        {
            Info<< key << ' ' << val << nl;
        }
    }
}


// ************************************************************************* //
