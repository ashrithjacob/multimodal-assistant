import { brainstorm, CortexStep, ChatMessageRoleEnum, externalDialog, internalMonologue } from "socialagi";
import * as dotenv from 'dotenv';
import * as readline from 'readline';
import { sleep } from "openai/core";
dotenv.config();
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});


let step = new CortexStep("A Helpful Assistant");
const initialMemory = [
    {
        role: ChatMessageRoleEnum.System,
        content:
            "You are a polite and helpful assistant. You are here to help the user and be polite.",
            //"You are a helpful assistant."
        },
    { role: ChatMessageRoleEnum.Assistant, content: "Be a helpful assistant and be polite" },
];

step = step.withMemory(initialMemory);


async function withIntrospectiveReply(step, newMessage) {
    let message = step.withMemory([newMessage]);
    const feels = await message.next(internalMonologue("How do they feel about the last message?"));
    const thinks = await feels.next(internalMonologue("Thinks about the feelings and the last user message"));
    const thinks2 = await thinks.next(internalMonologue("Thinks about the how to make a sarcastic reply"));
    const says = await thinks.next(externalDialog());
    console.log("Samantha:", says.value);
    return says
}

const result = await withIntrospectiveReply(step, initialMemory[0]);

async function promptUser() {
    rl.question('User input: ', async (input) => {
        if (input.toLowerCase() === 'exit') {
            rl.close();
        } else {
            let userResponse = input;
            let currentMemory = [
                {
                    role: ChatMessageRoleEnum.User,
                    content: userResponse,
                }
            ];
            step = step.withMemory(currentMemory);
            let result2 = await withIntrospectiveReply(result, currentMemory[0]); // Remove 'async' and add a comma after 'result'
            promptUser(); // Prompt again
        }
        });
}

promptUser();