"use client";
import React, { useEffect } from "react";
import { Navbar, Typography, IconButton } from "@material-tailwind/react";
import { Bars3Icon, XMarkIcon } from "@heroicons/react/24/outline";
import { useState } from "react";
import Image from "next/image";
import Logo from "../public/light-logo.png";
import Link from "next/link";

function NavList() {
  return (
    <ul className="my-2 flex flex-col items-end gap-2 lg:mb-0 lg:mt-0 lg:flex-row lg:items-center lg:gap-6">
      <Typography
        as="li"
        variant="small"
        color="white"
        className="p-1 pr-6 text-violet-50 font-medium"
      >
        <Link
          href="/about-us"
          className="flex items-center hover:text-violet-200 transition-colors"
        >
          About Us
        </Link>
      </Typography>

      <Typography
        as="li"
        variant="small"
        color="white"
        className="p-1 md:pr-0 md:mb-0 pr-6 mb-6 text-violet-50 font-medium"
      >
        <Link
          href="/sign-in"
          className="flex items-center hover:text-violet-200 transition-colors"
        >
          Login
        </Link>
      </Typography>
    </ul>
  );
}

export default function LandingNavBar() {
  const [openNav, setOpenNav] = useState(false);

  const handleWindowResize = () =>
    window.innerWidth >= 960 && setOpenNav(false);

  useEffect(() => {
    window.addEventListener("resize", handleWindowResize);

    return () => {
      window.removeEventListener("resize", handleWindowResize);
    };
  }, []);

  return (
    <Navbar className=" w-screen bg-neutral-800 px-6 py-3">
      <div className="flex items-center justify-between text-violet-50 px-5 py-5">
        <Link href="/">
          <Image width="70" height="70" src={Logo} alt="athena light logo" />
        </Link>
        <div className="hidden md:block">
          <NavList />
        </div>
        <IconButton
          variant="text"
          className="ml-auto h-6 w-6 text-violet-50 hover:bg-transparent focus:bg-transparent active:bg-transparent lg:hidden"
          ripple={false}
          onClick={() => setOpenNav(!openNav)}
        >
          {openNav ? (
            <XMarkIcon className="h-6 w-6" strokeWidth={2} />
          ) : (
            <Bars3Icon className="h-6 w-6" strokeWidth={2} />
          )}
        </IconButton>
      </div>
      {openNav && <NavList />}
    </Navbar>
  );
}
