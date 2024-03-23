<script setup lang="ts">
import { ref } from 'vue';
import MonacoEditor from 'monaco-editor-vue3';
import config from '@/config.json';

const title = ref(config.title);

const authors = ref(config.authors);

const res = ref(config.res);

const bibTex = ref(`@inproceedings{2024champ,
    author = {${authors.value.map(v => v.name).join(', ')}},
    title = {${title.value.title}: ${title.value.subtitle}},
    booktile = {arxiv}
    year = {2024}
}`);
</script>

<template>
  <main>
    <section class="title">
      <h1>{{ title.title }}</h1>
      <h3>{{ title.subtitle }}</h3>

      <div class="authors">
        <span v-for="author, i in authors" :key="i">
          <a :href="author.homepage" target="_blank">{{ author.name }}</a>
          <span v-if="i < authors.length - 1">, </span>
        </span>
      </div>

      <div class="res_link">
        <a class="button" :href="res.pdf" target="_blank">
          <i class="iconfont icon-lm-pdf"></i>
          <span>Paper</span>
        </a>
        <a class="button" :href="res.arxiv" target="_blank">
          <i class="iconfont icon-lm-Arxiv"></i>
          <span>arXiv</span>
        </a>

        <a class="button" :href="res.github" target="_blank">
          <i class="iconfont icon-lm-github"></i>
          <span>Code</span>
        </a>

        <a class="button" :href="res.huggingface" target="_blank">
          <i class="iconfont icon-lm-huggingface"></i>
          <span>HuggingFace</span>
        </a>
      </div>

      <video v-lazy src="@/assets/video/main_video.mp4" muted loop controls></video>
    </section>

    <section class="abstract">
      <div>
        <h3>Abstract</h3>
        <p>In this study, we introduce a methodology for human image animation by leveraging a 3D human parametric model
          within a latent diffusion framework to enhance shape alignment and motion guidance in curernt human generative
          techniques. The methodology utilizes the SMPL model as the 3D human parametric model to establish a unified
          representation of body shape and pose. This facilitates the accurate capture of intricate human geometry and
          motion characteristics from source videos. Specifically, we incorporate rendered depth images, normal maps,
          and
          semantic maps obtained from SMPL sequences, alongside skeleton-based motion guidance, to enrich the conditions
          to the latent diffusion model with comprehensive 3D shape and detailed pose attributes. A multi-layer motion
          fusion module, integrating self-attention mechanisms, is employed to fuse the shape and motion latent
          representations in the spatial domain. By representing the 3D human parametric model as the motion guidance,
          we
          can perform parametric shape alignment of the human body between the reference image and the source video
          motion. Experimental evaluations conducted on benchmark datasets demonstrate the methodology's superior
          ability
          to generate high-quality human animations that accurately capture both pose and shape variations. Furthermore,
          our approach also exhibits superior generalization capabilities on the proposed wild dataset. We will release
          our code and models for further research.</p>
      </div>
    </section>

    <section class="framework">
      <h3>Framework</h3>
      <div>
        <img src="@/assets/img/framework.jpg">
        <br>
        <p>Given an input human image and a reference video depicting a motion sequence, the objective is to synthesize
          a
          video where the person in the image replicates the actions observed in the reference video, thereby creating a
          controllable and temporally coherent visual output.</p>
      </div>

    </section>


    <section class="videos">
      <h3>Animate Human Image</h3>
      <div class="grid">
        <video v-lazy src="@/assets/video/human-animation/human-animation-1.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/human-animation/human-animation-2.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/human-animation/human-animation-3.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/human-animation/human-animation-4.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/human-animation/human-animation-5.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/human-animation/human-animation-6.mp4" muted loop controls></video>
      </div>
    </section>


    <section class="comparisions">
      <h3>Comparisions</h3>
      <div class="panel">
        <video v-lazy src="@/assets/video/comparisions/comparision-05.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/comparisions/comparision-07.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/comparisions/comparision-09.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/comparisions/comparision-10.mp4" muted loop controls></video>
      </div>
    </section>

    <section class="useen">
      <h3>Unseen Domain Animation</h3>
      <div class="grid">
        <video v-lazy src="@/assets/video/unseen/0.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/unseen/1.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/unseen/2.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/unseen/3.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/unseen/4.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/unseen/5.mp4" muted loop controls></video>
      </div>
    </section>


    <section class="t2i">
      <h3>Combining with T2I</h3>
      <div class="panel">
        <div>
          <video v-lazy src="@/assets/video/T2I/0.mp4" muted loop controls></video>
          <div class="t2i-caption">A woman in a silver dress posing for a picture, trending on cg society, futurism,
            with bright blue eyes,
            dior campaign, tesseract, miranda kerr --v 5. 1 --ar 3:4.</div>
        </div>

        <div>
          <video v-lazy src="@/assets/video/T2I/1.mp4" muted loop controls></video>
          <div class="t2i-caption">
            A realistic depiction of Aang, the last airbender, showcasing his mastery of all bending elements while in
            the powerful Avatar State. Ultra detailed, hd, 8k.
          </div>
        </div>

        <div>
          <video v-lazy src="@/assets/video/T2I/2.mp4" muted loop controls></video>
          <div class="t2i-caption">A painting of a woman in a yellow dress, heavy metal comic cover art, space theme,
            pin-up girl, silver
            and yellow color scheme, where the planets are candy, inspired by Joyce Ballantyne Brand, full - body
            artwork, lunar themed attire, golden age illustrator, blue and black color scheme.</div>
        </div>
      </div>
    </section>

    <section class="cross-id">
      <h3>Cross-ID Animation</h3>
      <div class="panel">
        <video v-lazy src="@/assets/video/cross-id/0.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/cross-id/1.mp4" muted loop controls></video>
        <video v-lazy src="@/assets/video/cross-id/2.mp4" muted loop controls></video>
      </div>
    </section>


    <section class="bibtex">
      <h3>BibTeX</h3>
      <div class="panel">
        <MonacoEditor class="editor" :value="bibTex" :height="130" language="json" theme="vs-dark"
          :options="{ renderValidationDecorations: 'off', readOnly: true, disableLayerHinting: true, hideCursorInOverviewRuler: true, onDidAttemptReadOnlyEdit: () => false, domReadOnly: true, automaticLayout: true, minimap: { enabled: false }, scrollbar: { vertical: 'hidden', handleMouseWheel: false } }">
        </MonacoEditor>
      </div>
    </section>

  </main>
</template>

<style scoped lang="scss">
main {
  @apply w-full h-full flex flex-col items-center my-5;

  h1 {
    @apply text-7xl text-center;
  }

  h2 {
    @apply text-5xl text-center my-8;
  }

  h3 {
    @apply text-4xl text-center my-5;
  }
}

section {
  @apply w-full py-5 md:px-16;
  @apply flex flex-col justify-center items-center;
}

.title {

  .authors {
    @apply text-center text-lg;
  }

  .res_link {
    @apply text-center mt-1;
  }

  video {
    max-width: 768px;
    @apply mt-4 block w-full;
  }
}

.framework {
  * {
    max-width: 768px;
    @apply w-full;
  }

}

.abstract {
  div {
    max-width: 768px;
    @apply w-full mt-2;
  }

  li {
    @apply flex flex-row my-1;

    :first-child {
      @apply mr-2;
    }
  }
}

.t2i-caption {
  @apply font-light italic md:px-20 text-center leading-snug;
}

.button {
  @apply mr-3 mt-2;

  i {
    @apply mr-1;
  }
}

.grid {
  max-width: 960px;
  @apply mt-2;
  @apply w-full flex flex-row justify-center items-center flex-wrap;

  &>* {
    @apply sm:w-1/2 sm:p-2;
  }
}

.panel {
  max-width: 960px;
  @apply w-full mt-2;

  &>* {
    @apply w-full mb-8;
  }

  :last-child {
    @apply mb-0;
  }
}

.editor {
  @apply min-w-full min-h-full;
}
</style>
