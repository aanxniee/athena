"use client";
import React from "react";
import Image from "next/image";
import LoginButton from "./LoginButton";
import Logo from "../public/dark.png";

export default function LandingFooter() {
  return (
    <>
      <div className="w-screen bg-violet-50 p-20 ">
        <div className="flex flex-col gap-5 md:mx-20">
          <Image width="200" height="200" src={Logo} alt="athena light logo" />
          <div className="text-4xl text-neutral-800">
            Sign up to get notified.
          </div>
          <div className="w-1/5">
            <LoginButton />
          </div>
        </div>
      </div>
      <div className="mx-auto max-w-screen-xl bg-neutral-800 px-6 py-3">
        <a
          href="/"
          className="flex items-center justify-center text-violet-50 hover:text-violet-200 px-5 py-5"
        >
          © Athena 2023
        </a>
      </div>
    </>
  );
}
