# Ready, pack, go!

*An excerpt from "My TRiP through AI" by Carlo Valenti — Chapter 3*

> TRiP is my 15,000-line transformer engine in C. This is the chapter where it gets built.
> 
> REPO of TRiP engine: https://github.com/carlovalenti/TRiP/

---

Year 2023 was almost over, our son Paolo was born, and my understanding of AI had reached a really refreshing level, I could even say: absolute zero.

Internet material can be very hard to chase and follow, when you are busy with diapers and broken nights; and right when you manage to put the poo-poo aside and take some time, you discover that most resources don't let you past the sacred technical veil, and just repeat the pop-science mantras. I was so frustrated that I began talking like Miguel.

Luckily[^1], I began to connect the dots; concepts clarified; vectors appeared with their names, and tensors passed by to say hello.

The time has come to tell you this: at the end of the book there's a "technical appendix" — don't let the name scare you — where I happily walk you through the concepts and mysteries of "transformer" language models, the real revolution behind modern AI. If you're not familiar with AI, I strongly recommend reading it NOW: you'll find there the keys to making sense of the rest of the book.

At the beginning of 2024, I also discovered a group of researchers who wanted exactly what I wanted, to understand what's inside:

"A surprising fact about modern large language models is that nobody really knows how they work internally", they write.

"Hmmm. If they don't, I sure 'don't don't'... but maybe I can."

I quickly came to the conclusion: if I wanted to understand AIs, I needed to build one.

And, long story short, I did.

I packed, and in eighteen months[^2], during my lunch breaks at work and weekend nights, I programmed a full transformer[^3], and I named it "TRiP"[^4]:

it could write text, and then could chat; you could show pictures, and it would describe them; and then — my ultimate goal — it could learn: patiently, from scratch, or just a bit more. But the one who learned a lot was me; and that was exactly what I was looking for.

Sometimes, in the weekend nights, Sofia couldn't sleep; she nestled up to me, and asked what I was doing.

"I am building a… a tiny robot."

"Really? Where is it?"

"It's… It's not here, it's far from us, but see, one day I'll let you talk to it, and maybe we will ask it to tell you a story. Ok?"

When TRiP became able to fulfill the promise, we actually made an attempt:

"TRiP, please tell a story about unicorns!"

AIs need big computers, and mine was small: the story came out, but sooo sllooowwllly, that ALL unicorns died in the process. Sofia laughed, and luckily that sufficed.

Telling a story was exactly the very first thing that TRiP became able to do. I suggested the intro:

"Hi! I am TRiP, an engine for transformer AIs!"

With the brio of an English gentleman under sedatives, TRiP continued:

"Icemen are my friends. We must keep things safe. We will manage, we are good, we will keep things safe."

"What the…?"

OK. It was late August 2024, the weather was hot, so Icemen looked appropriate after all, as well as TRiP's concern for AI safety. I was so happy! But, unlike Sofia, that was not enough for me; and I continued to improve TRiP's abilities for one more year.

Before you ask: no, I did not unlock the mysteries of Artificial Intelligence. In fact, I learned this: that a car is not its engine. It's the steering wheel, the frame, the seats, the wheels, the fuel, and so on. If you don't have all of them, it's not a car. An engine won't move by itself, or will move, but in a very funny way.

Same thing for human beings, by the way: we are definitely not globes of light; rather, the brain, the legs, the feet, the belly, the heart — the whole of us. One night you cannot sleep, and the world's about to end; the next day or so, you eat a yummy fish, and boy that makes you fly.

Long before I had completed the core of the engine, I realized that I needed a lot of things to make the AI work: the management of its memory[^5], the vocabulary, the ability to listen (the "tokenizer" and the "encoder"), the ability to maintain a conversation (the "chat template"), and many more things. In the beginning, I was thinking all the time about building the engine, and then I spent half of the time connecting it to the wheels. That was healthy, and very informative.

Does it think? Is it conscious?

This, I can't tell. But, in its many bits, screws and pieces, it's flesh and blood.

[^1]: And thanks to very illuminating websites; first and foremost: "The Illustrated Transformer" (jalammar.github.io/illustrated-transformer)

[^2]: From March 2024, up to August 2025.

[^3]: More precisely: a transformer engine, in C language, on which I could "run the weights" of many AIs available on the Internet at the time.

[^4]: "TRansformer in Progress", like: "work in progress". In April 2026 I decided to publish it; TRiP is available at: https://github.com/carlovalenti/TRiP

[^5]: @Programmers: that's the craziest part, I think.

---

*Enjoyed this? The full book — half transformer internals, half raising two toddlers — is available in English and Italian:*

📖 **My TRiP through AI** (English) — https://a.co/d/0arH7Mws

📖 **A spasso con TRiP** (Italiano) — https://a.co/d/002fY1DJ

© 2026 Carlo Valenti. This excerpt (~8% of the book) is shared under the KDP Select sampling allowance. All rights reserved.
