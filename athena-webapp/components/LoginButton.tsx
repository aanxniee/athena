"use client";

import Link from "next/link";

export default function LoginButton() {
  return (
    <Link
      href="sign-in"
      className="relative inline-block px-4 py-2 font-medium group"
    >
      <span className="absolute rounded-lg inset-0 w-full h-full transition duration-200 ease-out transform translate-x-1 translate-y-1 bg-neutral-800 group-hover:-translate-x-0 group-hover:-translate-y-0"></span>
      <span className="absolute rounded-lg inset-0 w-full h-full bg-violet-50 border-2 border-neutral-800 group-hover:bg-neutral-800"></span>
      <span className="relative text-xl text-neutral-800 group-hover:text-violet-50">
        Login
      </span>
    </Link>
  );
}
