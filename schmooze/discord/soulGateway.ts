import { Channel, Client, Events, Message, MessageType, ReplyOptions } from "discord.js";
import { ActionEvent, Soul } from "soul-engine/soul";
import { getMetadataFromActionEvent, makeMessageCreateDiscordEvent } from "./eventUtils.js";

export type DiscordEventData = {
  type: "messageCreate";
  messageId: string;
  channelId: string;
  guildId: string;
  userId: string;
  userDisplayName: string;
  atMentionUsername: string;
  repliedToUserId?: string;
};

export type DiscordAction = "chatted" | "joined";

export type SoulActionConfig =
  | {
      type: "says";
      sendAs: "message" | "reply";
    }
  | {
      type: "reacts";
      sendAs: "emoji";
    };

export class SoulGateway {
  private soul;
  private client;

  constructor(client: Client) {
    this.client = client;
    this.soul = new Soul({
      organization: process.env.SOUL_ENGINE_ORG!,
      blueprint: process.env.SOUL_BLUEPRINT!,
      soulId: process.env.SOUL_ID || undefined,
      token: process.env.SOUL_ENGINE_API_KEY || undefined,
      debug: process.env.SOUL_DEBUG === "true",
    });

    this.handleMessage = this.handleMessage.bind(this);
    this.onSoulSays = this.onSoulSays.bind(this);
  }

  start(readyClient: Client<true>) {
    const channel = this.client.channels.cache.get(process.env.DISCORD_CHANNEL_ID!);
    //console.log(channel)
    const pingTimeInSeconds = 10;
    let lastCheck = Date.now();
    console.log("lastCheck:", lastCheck);
    setInterval(() => this.checkNewMessage(channel, lastCheck), pingTimeInSeconds * 1000);
  }

  stop() {
    this.client.off(Events.MessageCreate, this.handleMessage);
    return this.soul.disconnect();
  }

  async onSoulSays(event: ActionEvent) {
    const { content } = event;
    const c = await content();
    console.log("soul said:", c);

    const { discordEvent, actionConfig } = getMetadataFromActionEvent(event);
    if (!discordEvent) return;

    console.log("soul said something");

    let reply: ReplyOptions | undefined = undefined;
    if (discordEvent.type === "messageCreate" && actionConfig?.sendAs === "reply") {
      reply = {
        messageReference: discordEvent.messageId,
      };
    }

    const channel = await this.client.channels.fetch(process.env.DISCORD_CHANNEL_ID!);
    if (channel && channel.isTextBased()) {
      await channel.sendTyping();
      channel.send({
        content: await content(),
        reply,
      });
    }
  }

  async handleMessage(discordMessage: Message) {
    const messageSenderIsBot = !!discordMessage.author.bot;
    const messageSentInCorrectChannel = discordMessage.channelId === process.env.DISCORD_CHANNEL_ID;
    const shouldIgnoreMessage = messageSenderIsBot || !messageSentInCorrectChannel;
    if (shouldIgnoreMessage) {
      return;
    }

    const timestamp = discordMessage.createdAt.getTime();
    const t = typeof timestamp;
    console.log('Message timestamp:', timestamp, t);
    const currentUTCTime = new Date().getTime();
    console.log(currentUTCTime);

    const discordEvent = await makeMessageCreateDiscordEvent(discordMessage);
    const userName = discordEvent.atMentionUsername;
    console.log("discordEvent:", discordEvent);
    console.log("userName:", userName);

    const userJoinedSystemMessage = discordMessage.type === MessageType.UserJoin;
    if (userJoinedSystemMessage) {
      this.soul.dispatch({
        action: "joined",
        content: `${userName} joined the server`,
        name: userName,
        _metadata: {
          discordEvent,
          botUserId: this.client.user?.id,
        },
      });
      return;
    }

    let content = discordMessage.content;
    console.log("content:", content);
    if (discordEvent.repliedToUserId) {
      content = `<@${discordEvent.repliedToUserId}> ${content}`;
    }
    /*
    this.soul.dispatch({
      action: "chatted",
      content,
      name: userName,
      _metadata: {
        discordEvent,
        botUserId: this.client.user?.id,
      },
    });
    */
    
    const channel = await this.client.channels.fetch(process.env.DISCORD_CHANNEL_ID!);
    if (channel && channel.isTextBased()) {
      await channel.sendTyping();
    }
  }

  async checkNewMessage(channel, lastCheck) {
    console.log("In function", lastCheck);
    // Your function logic goes here
    const messages = await channel.messages.fetch({ limit: 10 });
    messages.forEach((message) => {
      // Check if the message was sent after the last check
      if (message.createdTimestamp > lastCheck) {
        //console.log(`New message from ${message.author.username}: ${message.content}`);
        console.log(`Message timestamp: ${message.createdTimestamp}\t lastCheck: ${lastCheck}`);
      }
      else {
        console.log(`Old message`);
      }
    });
    lastCheck = Date.now();
  }
}
