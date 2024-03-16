<script setup lang="ts">
import { ref } from 'vue';
import MonacoEditor from 'monaco-editor-vue3';

const authors = ref([
  {
    name: 'Xiaoming Wang',
    link: 'https://github.com/leeway213',
  },
  {
    name: 'Lily Zhang',
    link: 'https://github.com/leeway213',
  },
  {
    name: 'Leeway Zhang',
    link: 'https://github.com/leeway213',
  },
  {
    name: 'Leeway Zhang',
    link: 'https://github.com/leeway213',
  },
  {
    name: 'Leeway Zhang',
    link: 'https://github.com/leeway213',
  },
  {
    name: 'Leeway Zhang',
    link: 'https://github.com/leeway213',
  }
]);

const bibTex = ref(`@inproceedings{xu2023magicanimate,
    author = {Xu, Zhongcong and Zhang, Jianfeng and Liew, Jun Hao and Yan, Hanshu and Liu, Jia-Wei and Zhang, Chenxu and Feng, Jiashi and Shou, Mike Zheng},
    title = {MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model},
    booktile = {arxiv}
    year = {2023}
}`);
</script>

<template>
  <main>
    <h1>CHAMP</h1>
    <br>
    <h2>Controllable and Consistent Human Image Animation with 3D Parametric (SMPL) </h2>

    <br>
    <div class="authors">
      <span v-for="author, i in authors" :key="i">
        <a :href="author.link" target="_blank">{{ author.name }}</a>
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
      <video src="https://box.nju.edu.cn/seafhttp/files/92d85d35-e7ba-4b54-99c6-1dd52cf8966f/main_video.mp4" autoplay
        muted loop controls></video>
    </div>

    <br>
    <br>
    <h2>Video Result</h2>
    <br>
    <h3>Animate Human Image</h3>
    <div class="grid">
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
    </div>

    <br>
    <h3>Comparsions</h3>
    <div class="grid">
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
    </div>

    <br>
    <h3>Cross ID</h3>
    <div class="panel">
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
    </div>

    <br>
    <h3>Unseen domain</h3>
    <div class="grid">
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
    </div>

    <br>
    <h3>Combining with T2I</h3>
    <div class="grid">
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
      <video src="https://showlab.github.io/magicanimate/assets/teaser/t1.mp4" autoplay muted loop></video>
    </div>

    <br>
    <br>
    <h3>Pipeline</h3>
    <div class="panel">
      <img src="https://box.nju.edu.cn/seafhttp/files/1e6d3b00-c696-4bb6-8600-ffe3c5edfb5e/framework.jpg">
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
    @apply text-7xl;
  }

  h2 {
    @apply text-4xl;
  }

  h3 {
    @apply text-2xl;
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
    @apply mt-2;
  }
}

.editor {
  @apply min-w-full min-h-full;
}
</style>
