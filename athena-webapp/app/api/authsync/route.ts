import { prisma } from "@/lib/prisma";
import { NextResponse } from "next/server";

export async function POST(req: Request, res: Response) {
  try {
    let body = await req.json();
    body = body.data;
    console.log("request object:", body);

    //create user
    await prisma.user.create({
      data: {
        userId: body.id,
        firstName: body.first_name,
        lastName: body.last_name,
        email: body.email_addresses[0].email_address,
        gradYear: -1,
      },
    });
    return NextResponse.json({text: "Printed"})
  } catch (error) {
    console.error("Error:", error);
    return new Response('Error', {
      status: 500,
    });  
  } 
}