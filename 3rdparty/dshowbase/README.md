*dshowbase* Module
------------------
This directory contains the [DirectShow base classes](https://msdn.microsoft.com/en-us/library/windows/desktop/dd375456(v=vs.85).aspx) as previously distributed in the Windows SDK v7.1, along with some new utilities. The DirectShow base classes are required to build, deploy and register DirectShow filters for multimedia capture and preview on Windows platforms. These were packaged as part of the Windows SDK samples for years, but lately, it is getting harder and harder to find them online (especially since the Windows 10 launch). These are provided here under Microsoft's original SDK samples licensing terms; please refer to the LICENSE.htm file (as distributed with the Windows SDK v7.1) for more information.

This project is inspired by Leapmotion's similar initiative; see:
<https://github.com/leapmotion/DShowBaseClasses/>

The GMFBridge utility was introduced by Geraint Davies; for more information, see:
<http://www.gdcl.co.uk/gmfbridge/>

A noteworthy addition to this module is [ocvcompat.h](./include/litiv/3rdparty/dshowbase/ocvcompat.h): this file defines an object wrapper for a DirectShow graph which simplifies video device connection and frame acquisition for capture applications.
