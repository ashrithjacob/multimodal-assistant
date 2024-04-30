import { Soul, said } from "soul-engine/soul"

// Create a new Soul instance with a new unique identifier
const soul = new Soul({
  organization: "mintermute2000",
  blueprint: "schmoozie",
  token: "47c9bf34-55a5-4629-bc58-a6abc0191bd9", // npx soul-engine apikey
  debug: true, // this is new
})

 
// Set up a listener for when the soul speaks
soul.on("says", async ({ content }) => {
  console.log("Samantha said", await content())
})
 
// Connect the soul to the engine
soul.connect().then(async () => {
  // Interact with the soul
  soul.dispatch(said("User", "Hi Schoomzie!"))
})