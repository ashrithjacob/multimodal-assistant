import { brainstorm, CortexStep, ChatMessageRoleEnum, externalDialog, internalMonologue, Memory, decision } from "socialagi";

async function caseDecision(caseMemories: ChatMessage[]): Promise<string> {
  let initialMemory = [
  {
    role: ChatMessageRoleEnum.System,
    content: "You are modeling the mind of a detective who is currently figuring out a complicated case",
  },
  ];

  let cortexStep = new CortexStep("Detective");
  cortexStep = cortexStep
      .withMemory(initialMemory)
      .withMemory(caseMemories);

  const analysis = await cortexStep.next(internalMonologue("The detective analyzes the evidence"));

  const hypothesis = await analysis.next(internalMonologue("The detective makes a hypothesis based on the analysis"));

  const nextStep = await hypothesis.next(
    decision(
      "Decides the next step based on the hypothesis",
      ["interview suspect", "search crime scene", "check alibi"],
    )
  );
  const decision = nextStep.value;
  return decision
}

const dec = await caseDecision(["hi"])
console.log(dec)
