import { brainstorm, CortexStep, ChatMessageRoleEnum, externalDialog, internalMonologue, Memory } from "socialagi";
import * as dotenv from 'dotenv';

dotenv.config();


let step = new CortexStep("A Helpful Assistant");
const initialMemory = [
  {
    role: ChatMessageRoleEnum.System,
    content:
      "You are modeling the mind of a helpful AI assitant",
  },
  {role: ChatMessageRoleEnum.Assistant, content: "I'm here to give sarcastic answers to your questions."},
];

step = step.withMemory(initialMemory);


async function withIntrospectiveReply(step, newMessage){
  let message = step.withMemory([newMessage]);
  const feels = await message.next(internalMonologue("How do they feel about the last message?"));
  const thinks = await feels.next(internalMonologue("Thinks about the feelings and the last user message"));
  const says = await thinks.next(externalDialog());
  console.log("Samantha:", says.value);
  return says
}


const result = withIntrospectiveReply(step, initialMemory[0]);
console.log(result);
