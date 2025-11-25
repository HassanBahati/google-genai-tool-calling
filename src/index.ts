import { googleAI, vertexAI } from "@genkit-ai/google-genai";
import { genkit, z } from "genkit";

const ai = genkit({
  plugins: [googleAI(), vertexAI({ location: "global" })],
  model: googleAI.model("gemini-2.5-flash", {
    temperature: 0.8,
  }),
});

export const vertexAIExplanationFlow = ai.defineFlow(
  {
    name: "vertexAIExplanationFlow",
  },
  async (input) => {
    const response = await ai.generate({
      model: vertexAI.model("gemini-3-pro-preview"),
      prompt: input.prompt || "Explain Vertex AI in simple terms.",
    });

    return {
      text: response.text || "",
    };
  }
);


export const vertexAIExplanationImageFlow = ai.defineFlow(
  {
    name: "vertexAIExplanationImageFlow",
  },
  async (input) => {
    const response = await ai.generate({
      model: vertexAI.model("gemini-3-pro-image-preview"),
      prompt: input.prompt || "Explain the image in simple terms.",
    });

    return {
      text: response.text || "",
    };
  }
);

ai.defineFlow("list-models", async () => {
  const plugin = googleAI();

  if (!plugin.list) {
    throw new Error("Plugin does not have a list method");
  }

  const listedModels = await plugin.list();

  // Helper function to extract model info
  const getModelInfo = (model: { name: string }) => {
    const nameParts = model.name.split("/");
    const modelId = nameParts[nameParts.length - 1] || "";

    return {
      label: modelId,
      versions: [],
      supports: {
        text: true,
        image: modelId.includes("vision") || modelId.includes("flash"),
        audio: false,
      },
    };
  };

  return {
    listedModelsCount: listedModels.length,
    allListedModels: listedModels.map((m) => {
      const modelInfo = getModelInfo(m);
      return {
        name: m.name,
        label: modelInfo?.label || "N/A",
        versions: modelInfo?.versions || [],
        supports: modelInfo?.supports || {},
      };
    }),
  };
});

// Define input schema
const RecipeInputSchema = z.object({
  ingredient: z.string().describe("Main ingredient or cuisine type"),
  dietaryRestrictions: z
    .string()
    .optional()
    .describe("Any dietary restrictions"),
});

// Define output schema
const RecipeSchema = z.object({
  title: z.string(),
  description: z.string(),
  prepTime: z.string(),
  cookTime: z.string(),
  servings: z.number(),
  ingredients: z.array(z.string()),
  instructions: z.array(z.string()),
  tips: z.array(z.string()).optional(),
});

export const getWeather = ai.defineTool(
  {
    name: "getWeather",
    description: "Gets the current weather in a given location",
    inputSchema: z.object({
      location: z
        .string()
        .describe("The location to get the current weather for"),
    }),
    outputSchema: z.object({
      temperature: z
        .number()
        .describe("The current temperature in degrees Fahrenheit"),
      condition: z
        .enum(["sunny", "cloudy", "rainy", "snowy"])
        .describe("The current weather condition"),
    }),
  },
  async ({ location }) => {
    const randomTemp = Math.floor(Math.random() * 30) + 50; // Random temp between 50 and 80
    const conditions: Array<"sunny" | "cloudy" | "rainy" | "snowy"> = [
      "sunny",
      "cloudy",
      "rainy",
      "snowy",
    ];
    const randomCondition =
      conditions[Math.floor(Math.random() * conditions.length)]!;
    return { temperature: randomTemp, condition: randomCondition };
  }
);

// Define weather input schema
const WeatherInputSchema = z.object({
  location: z.string().describe("The location to get the weather for"),
});

// Define weather output schema
const WeatherOutputSchema = z.object({
  location: z.string(),
  temperature: z.number(),
  condition: z.string(),
  message: z.string(),
});

// Define a weather flow that uses the getWeather tool
export const weatherFlow = ai.defineFlow(
  {
    name: "weatherFlow",
    inputSchema: WeatherInputSchema,
    outputSchema: WeatherOutputSchema,
  },
  async (input) => {
    // Call the weather tool directly
    const weather = await getWeather({ location: input.location });

    const { text } = await ai.generate({
      prompt: `The weather in ${input.location} is ${weather.condition} with a temperature of ${weather.temperature}°F. Create a friendly, conversational message about this weather.`,
    });

    return {
      location: input.location,
      temperature: weather.temperature,
      condition: weather.condition,
      message:
        text ||
        `The weather in ${input.location} is ${weather.condition} with a temperature of ${weather.temperature}°F.`,
    };
  }
);

// Define a recipe generator flow
export const recipeGeneratorFlow = ai.defineFlow(
  {
    name: "recipeGeneratorFlow",
    inputSchema: RecipeInputSchema,
    outputSchema: RecipeSchema,
  },
  async (input) => {
    // Create a prompt based on the input
    const prompt = `Create a recipe with the following requirements:
      Main ingredient: ${input.ingredient}
      Dietary restrictions: ${input.dietaryRestrictions || "none"}`;

    // Generate structured recipe data using the same schema
    const { output } = await ai.generate({
      prompt,
      output: { schema: RecipeSchema },
    });

    if (!output) throw new Error("Failed to generate recipe");

    return output;
  }
);

async function exampleWeatherChat() {
  const response = await ai.generate({
    prompt: "What is the weather like in San Francisco?",
    tools: [getWeather],
  });

  console.log("Response:", response.text);
  // call getWeather({ location: 'San Francisco' })
}

// Run the flow
async function main() {
  // Vertex AI explanation example
  const vertexAIResponse = await vertexAIExplanationFlow({
    prompt: "Explain Vertex AI in simple terms.",
  });
  console.log("Vertex AI Response:", vertexAIResponse.text);

  // Examples
  const recipe = await recipeGeneratorFlow({
    ingredient: "avocado",
    dietaryRestrictions: "vegetarian",
  });

  console.log("Recipe:", recipe);

  console.log("\n--- Weather Chat Example ---");
  await exampleWeatherChat();
}

main().catch(console.error);
