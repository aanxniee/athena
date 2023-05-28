import LandingProcess from "@/components/LandingProcess";
import LoginButton from "@/components/LoginButton";
import Image from "next/image";
import Logo from "../../public/dark.png";
export default function Home() {
  return (
    <>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" />
      <link
        href="https://fonts.googleapis.com/css2?family=Poppins&display=swap"
        rel="stylesheet"
      />
      <main className="flex min-h-screen flex-col items-center justify-between p-24">
        <div className="z-10 w-full max-w-5xl items-center justify-between text-sm text-violet-50 my-20 lg:flex">
          <div className="flex md:flex-row flex-col justify-center items-center">
            <Image
              width="500"
              height="500"
              src={Logo}
              alt="athena light logo"
            />
            <div className="flex-col">
              <div className="md:text-8xl text-6xl pt-3 font-bold text-shadow">
                Athena
              </div>
              <div className="md:text-xl text-lg py-4">
                this is a placeholder
                <br />
                this is a placeholder
                <br />
                this is a placeholder
              </div>
              <LoginButton />
            </div>
          </div>
        </div>
        <div className="w-screen md:w-[80vw] md:m-10 md:py-20 p-14 text-neutral-800 font-bold flex md:flex-row flex-col justify-center items-center">
          <div className="text-2xl md:w-1/2">
            Comprehensive questionnaire to make the{" "}
            <span className="text-violet-500">BEST</span> match.
          </div>
          <div className="text-xl md:w-1/2 md:text-right py-5">
            <ul>
              <li>this is a placeholder</li>
              <li>this is a placeholder</li>
              <li>this is a placeholder</li>
              <li>this is a placeholder</li>
            </ul>
          </div>
        </div>
        <div className="w-screen text-neutral-800 flex flex-col justify-center items-center">
          <div className="text-4xl font-bold py-10">Process</div>
          <LandingProcess />
        </div>
      </main>
    </>
  );
}
