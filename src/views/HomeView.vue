<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import config from '@/config.json';
import ComparisionVideos from '@/components/ComparisionVideos.vue';
import AnimateHumanVideos from '@/components/AnimateHumanVideos.vue';
import UnseenVideos from '@/components/UnseenVideos.vue';
import T2IVideos from '@/components/T2IVideos.vue';
import CrossIdVideos from '@/components/CrossIdVideos.vue';

const title = ref(config.title);

const authors = ref(config.authors);

const res = ref(config.res);

const bibTex = ref(`@misc{zhu2024champ,
      title={Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance}, 
      author={Shenhao Zhu and Junming Leo Chen and Zuozhuo Dai and Yinghui Xu and Xun Cao and Yao Yao and Hao Zhu and Siyu Zhu},
      year={2024},
      eprint={2403.14781},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
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

        <!-- <a class="button" :href="res.huggingface" target="_blank">
          <i class="iconfont icon-lm-huggingface"></i>
          <span>HuggingFace</span>
        </a> -->
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


    <section class="useen">
      <h3>Unseen Domain Animation</h3>
      <div class="panel">
        <UnseenVideos></UnseenVideos>
      </div>
    </section>

    <section class="cross-id">
      <h3>Cross-ID Animation</h3>
      <div class="panel">
        <CrossIdVideos></CrossIdVideos>
      </div>
    </section>

    <section class="t2i">
      <h3>Combining with T2I</h3>
      <div class="panel">
        <T2IVideos></T2IVideos>
      </div>
    </section>

    <section class="comparisions">
      <h3>Comparisions with Existed Approaches</h3>
      <div class="panel">
        <ComparisionVideos></ComparisionVideos>
      </div>
    </section>


    <section class="videos">
      <h3>Animation on TikTok Dataset</h3>
      <div class="panel">
        <AnimateHumanVideos></AnimateHumanVideos>
      </div>
    </section>


    <section class="bibtex">
      <h3>BibTeX</h3>
      <pre class="bibtex-code"><code>{{ bibTex }}</code></pre>
    </section>

  </main>
</template>

<style scoped lang="scss">
main {
  @apply w-full h-full flex flex-col items-center;

  >:nth-child(2n-1) {
    @apply bg-white;
    @apply dark:bg-black/50;
  }

  h1 {
    @apply text-7xl text-center;
  }

  h2 {
    @apply text-5xl text-center my-8;
  }

  h3 {
    @apply text-3xl text-center my-5;
  }
}

section {
  @apply w-full py-10 md:px-16 px-6;
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
    max-width: 960px;
    @apply mt-4 block w-full;
  }
}

.framework {
  * {
    max-width: 960px;
    @apply w-full;
  }

}

.abstract {
  div {
    max-width: 960px;
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

.bibtex-code {
  @apply border-gray-300 bg-gray-300/15 p-4 rounded-lg w-full overflow-auto;
  max-width: 960px;
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

.carousel__item {
  min-height: 200px;
  width: 100%;
  background-color: var(--vc-clr-primary);
  color: var(--vc-clr-white);
  font-size: 20px;
  border-radius: 8px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.carousel__slide {
  padding: 10px;
}

.carousel__prev,
.carousel__next {
  box-sizing: content-box;
  border: 5px solid white;
}
</style>
