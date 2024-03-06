//import { CortexStep, decision, externalDialog } from "socialagi";
//import { MentalProcess, useActions, usePerceptions, useSoulMemory } from "soul-engine";
//import { Perception } from "soul-engine/soul";

import { Soul, said } from "soul-engine/soul";

// Create a new Soul instance with a new unique identifier
const soul = new Soul({
    organization: process.env.SOUL_ENGINE_ORG,
    blueprint: process.env.SOUL_BLUEPRINT,
    soulId: process.env.SOUL_ID || undefined,
    token: process.env.SOUL_ENGINE_API_KEY || undefined,
    debug: process.env.SOUL_DEBUG === "true",
});

// Register a listener for "says" interaction requests from the soul
soul.on("says", async ({ stream }) => {
    // Stream is a function returning an AsyncIterable<string>
    for await (const message of stream()) {
        // Process each message chunk
        console.log("Samantha uttered (chunk):", message)
    }
})

// Connect the soul to the engine
soul.connect().then(async () => {
    // Send a greeting to the soul
    soul.dispatch(said("User", "Hi!"))
})