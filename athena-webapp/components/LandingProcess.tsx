"use client";
import { Fragment, useState } from "react";
import {
  Accordion,
  AccordionHeader,
  AccordionBody,
} from "@material-tailwind/react";

export default function Example() {
  const [openA1, setOpenA1] = useState(false);
  const [openA2, setOpenA2] = useState(false);
  const [openA3, setOpenA3] = useState(false);

  const handleOpen = (setFunction: any, value: boolean) => {
    setFunction(!value);
  };

  return (
    <Fragment>
      <Accordion
        open={openA1}
        onClick={() => handleOpen(setOpenA1, openA1)}
        className="w-[72%] bg-violet-50 border-neutral-800 border-2 box-shadow p-6 px-10 rounded-lg mb-10"
      >
        <AccordionHeader
          className={`border-b-0 transition-colors ${
            openA1 ? "text-neutral-800 hover:!text-violet-500" : ""
          }`}
        >
          this is a placeholder
        </AccordionHeader>
        <AccordionBody className="text-base font-normal pt-0">
          {openA1 && <div>this is another placeholder</div>}
        </AccordionBody>
      </Accordion>

      <Accordion
        open={openA2}
        onClick={() => handleOpen(setOpenA2, openA2)}
        className="w-[72%] bg-violet-50 border-neutral-800 border-2 box-shadow p-6 px-10 rounded-lg mb-10"
      >
        <AccordionHeader
          className={`border-b-0 transition-colors ${
            openA2 ? "text-neutral-800 hover:!text-violet-500" : ""
          }`}
        >
          this is a placeholder
        </AccordionHeader>
        <AccordionBody className="text-base font-normal pt-0">
          {openA2 && <div>this is another placeholder</div>}
        </AccordionBody>
      </Accordion>

      <Accordion
        open={openA3}
        onClick={() => handleOpen(setOpenA3, openA3)}
        className="w-[72%] bg-violet-50 border-neutral-800 border-2 box-shadow p-6 px-10 rounded-lg mb-10"
      >
        <AccordionHeader
          className={`border-b-0 transition-colors ${
            openA3 ? "text-neutral-800 hover:!text-violet-500" : ""
          }`}
        >
          this is a placeholder
        </AccordionHeader>
        <AccordionBody className="text-base font-normal pt-0">
          {openA3 && <div>this is another placeholder</div>}
        </AccordionBody>
      </Accordion>
    </Fragment>
  );
}
