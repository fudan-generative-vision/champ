<script setup lang="ts">
import { ref } from 'vue';
import MonacoEditor from 'monaco-editor-vue3';
import config from '@/config.json';

const title = ref(config.title);

const authors = ref(config.authors);

const bibTex = ref(`@inproceedings{zhu2024champ,
    author = {${authors.value.map(v => v.name).join(', ')}},
    title = {${title.value.title}: ${title.value.subtitle}},
    booktile = {arxiv}
    year = {2024}
}`);
</script>

<template>
  <main>
    <h1>{{ title.title }}</h1>
    <br>
    <h2>{{ title.subtitle }}</h2>

    <br>
    <div class="authors">
      <span v-for="author, i in authors" :key="i">
        <a :href="author.homepage" target="_blank">{{ author.name }}</a>
        <span v-if="i < authors.length - 1">, </span>
      </span>
    </div>

    <br>
    <div class="res_link">
      <a class="button">
        <i class="iconfont icon-lm-pdf"></i>
        <span>Paper</span>
      </a>
      <a class="button">
        <i class="iconfont icon-lm-Arxiv"></i>
        <span>arXiv</span>
      </a>

      <a class="button">
        <i class="iconfont icon-lm-github"></i>
        <span>Code</span>
      </a>

      <a class="button">
        <i class="iconfont icon-lm-huggingface"></i>
        <span>HuggingFace</span>
      </a>
    </div>

    <br>
    <div class="main_video">
      <video v-lazy src="@/assets/video/main_video.mp4" muted loop controls></video>
    </div>

    <br>
    <br>
    <div class="abstract">
      <h2>Abstract</h2>
      <p>In this study, we introduce a methodology for human image animation by leveraging a 3D human parametric model
        within a latent diffusion framework to enhance shape alignment and motion guidance in curernt human generative
        techniques. The methodology utilizes the SMPL model as the 3D human parametric model to establish a unified
        representation of body shape and pose. This facilitates the accurate capture of intricate human geometry and
        motion characteristics from source videos. Specifically, we incorporate rendered depth images, normal maps, and
        semantic maps obtained from SMPL sequences, alongside skeleton-based motion guidance, to enrich the conditions
        to the latent diffusion model with comprehensive 3D shape and detailed pose attributes. A multi-layer motion
        fusion module, integrating self-attention mechanisms, is employed to fuse the shape and motion latent
        representations in the spatial domain. By representing the 3D human parametric model as the motion guidance, we
        can perform parametric shape alignment of the human body between the reference image and the source video
        motion. Experimental evaluations conducted on benchmark datasets demonstrate the methodology's superior ability
        to generate high-quality human animations that accurately capture both pose and shape variations. Furthermore,
        our approach also exhibits superior generalization capabilities on the proposed wild dataset. We will release
        our code and models for further research.</p>
      <br>
      <h3>Framework</h3>
      <p>Given an input human image and a reference video depicting a motion sequence, the objective is to synthesize a
        video where the person in the image replicates the actions observed in the reference video, thereby creating a
        controllable and temporally coherent visual output.</p>
      <br>
      <h3>T2I</h3>
      <p>
      <ul>
        <li><span>1. </span><span>A woman in a silver dress posing for a picture, trending on cg society, futurism, with
            bright blue eyes,
            dior campaign, tesseract, miranda kerr --v 5. 1 --ar 3:4.</span></li>
        <li><span>2. </span> <span>A realistic depiction of Aang, the last airbender, showcasing his mastery of all
            bending elements while
            in
            the powerful Avatar State. Ultra detailed, hd, 8k.</span></li>
        <li><span>3. </span>
          <span>A painting of a woman in a yellow dress, heavy metal comic cover art, space theme, pin-up girl, silver
            and
            yellow color scheme, where the planets are candy, inspired by Joyce Ballantyne Brand, full - body artwork,
            lunar themed attire, golden age illustrator, blue and black color scheme.</span>
        </li>
      </ul>
      </p>
    </div>

    <br>
    <br>
    <h2>Video Result</h2>
    <br>
    <h3>Animate Human Image</h3>
    <div class="grid">
      <video v-lazy src="@/assets/video/human-animation/human-animation-1.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/human-animation/human-animation-2.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/human-animation/human-animation-3.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/human-animation/human-animation-4.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/human-animation/human-animation-5.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/human-animation/human-animation-6.mp4" muted loop controls></video>
    </div>

    <br>
    <h3>Comparsions</h3>
    <div class="panel">
      <video v-lazy src="@/assets/video/comparisions/comparision-05.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/comparisions/comparision-07.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/comparisions/comparision-09.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/comparisions/comparision-10.mp4" muted loop controls></video>
    </div>

    <br>
    <h3>Cross-ID Animation</h3>
    <div class="panel">
      <video v-lazy src="@/assets/video/cross-id/0.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/cross-id/1.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/cross-id/2.mp4" muted loop controls></video>
    </div>

    <br>
    <h3>Unseen Domain Animation</h3>
    <div class="grid">
      <video v-lazy src="@/assets/video/unseen/0.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/unseen/1.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/unseen/2.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/unseen/3.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/unseen/4.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/unseen/5.mp4" muted loop controls></video>
    </div>

    <br>
    <h3>Combining with T2I</h3>
    <div class="panel">
      <video v-lazy src="@/assets/video/T2I/0.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/T2I/1.mp4" muted loop controls></video>
      <video v-lazy src="@/assets/video/T2I/2.mp4" muted loop controls></video>
    </div>

    <br>
    <br>
    <h3>Framework</h3>
    <div class="panel">
      <img src="@/assets/img/framework.jpg">
    </div>

    <br>
    <br>
    <h3>BibTeX</h3>
    <div class="panel">
      <MonacoEditor class="editor" :value="bibTex" :height="130" language="json" theme="vs-dark"
        :options="{ renderValidationDecorations: 'off', readOnly: true, disableLayerHinting: true, hideCursorInOverviewRuler: true, onDidAttemptReadOnlyEdit: () => false, domReadOnly: true, automaticLayout: true, minimap: { enabled: false }, scrollbar: { vertical: 'hidden', handleMouseWheel: false } }">
      </MonacoEditor>
    </div>
  </main>
</template>

<style scoped lang="scss">
main {
  @apply block w-full h-full flex flex-col items-center my-5;

  h1 {
    @apply text-7xl text-center;
  }

  h2 {
    @apply text-4xl text-center leading-loose;
  }

  h3 {
    @apply text-2xl text-center leading-loose;
  }
}

.button {
  @apply mr-3;

  i {
    @apply mr-1;
  }
}

.main_video {
  max-width: 720px;
  @apply w-full;
}

.abstract {
  max-width: 720px;

  p {
    @apply leading-relaxed;
  }

  li {
    @apply flex flex-row;

    :first-child {
      @apply mr-2;
    }
  }
}

.grid {
  max-width: 960px;
  @apply mt-2;
  @apply w-full flex flex-row justify-center items-center flex-wrap;

  &>* {
    @apply w-1/2 p-2;
  }
}

.panel {
  max-width: 960px;
  @apply w-full mt-2;

  &>* {
    @apply mb-8;
  }
}

.editor {
  @apply min-w-full min-h-full;
}
</style>
