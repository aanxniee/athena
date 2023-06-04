import React from "react";
import Image from "next/image";
import Logo from "../../../public/headshots/Me.png";

export default function Page() {
  return (
    <>
      <main className="flex min-h-screen flex-col justify-between p-24">
        <div className="z-10 w-full max-w-5xl text-sm text-violet-50 my-20 lg:flex">
          <div className="flex md:flex-row flex-col">
            <div className="flex-col">
              <div className="md:text-8xl text-6xl pt-3 font-bold text-shadow">
                About Us
              </div>
              <div className="pt-40 text-2xl font-bold text-neutral-800">
                About The Project
              </div>
              <p className="pt-4 text-neutral-800">
                Every year, first year System Design Engineering (SYDE) students
                at the University of Waterloo gets paired with an Upper Year
                SYDE Mentor to help them to learn the ropes and get advice. The
                mentor-mentee program is unique to SYDEs and though many SYDEs
                get to enjoy this program, the process of creating these pairs
                is time-consuming and the matching could be improved.
                <br></br>
                Identifying the opportunity to reduce manual matching, optimize
                matches, and create a platform to encourage even more
                interactions between SYDEs of all years, we decided to create
                Athena.
              </p>
              <div className="pt-40 text-2xl font-bold text-neutral-800">
                About The Team
              </div>
              <p className="pt-4 italic text-neutral-800">
                Developed by SYDEs for SYDEs
              </p>
              <p className="text-neutral-800">
                Athena is developed by four SYDE 27s who want to further bolster
                the connections between all SYDEs
              </p>
              <div className="flex pt-12 pb-24 space-x-8 items-start">
                <Image
                  width="200"
                  height="200"
                  src={Logo}
                  alt="athena light logo"
                  className="rounded-full"
                />
                <Image
                  width="200"
                  height="200"
                  src={Logo}
                  alt="athena light logo"
                  className="rounded-full"
                />
                <Image
                  width="200"
                  height="200"
                  src={Logo}
                  alt="athena light logo"
                  className="rounded-full"
                />
                <Image
                  width="200"
                  height="200"
                  src={Logo}
                  alt="athena light logo"
                  className="rounded-full"
                />
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
