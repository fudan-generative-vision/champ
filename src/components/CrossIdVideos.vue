<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import { initTWE, Carousel } from 'tw-elements';
import { inVisible } from '@/utils/video';
onMounted(() => {
  initTWE({ Carousel }, { allowReinits: true });

  crossIdCarousel.value?.addEventListener('slide.twe.carousel', (v: any) => {
    const from = v.from;
    const to = v.to;
    videos.value[2 * from]?.pause();
    videos.value[2 * from + 1]?.pause();
    if (inVisible(videos.value[2 * from])) {
      videos.value[2 * to].play();
      videos.value[2 * to + 1].play();
    }
  })
});

const crossIdCarousel = ref<HTMLElement>();
const videos = ref<HTMLVideoElement[]>([]);
// watch(crossIdCarousel, (newV) => {
//   if (newV) {
//     newV.addEventListener('slide.twe.carousel', (v: any) => {
//       const from = v.from;
//       const to = v.to;
//       videos.value[from]?.pause();
//       videos.value[from + 1]?.pause();
//       videos.value[2 * to]?.play();
//       videos.value[2 * to + 1]?.play();
//     })
//   }
// }, { once: true });
</script>

<template>
  <div ref="crossIdCarousel" id="crossIdCarousel" class="relative" data-twe-carousel-init data-twe-ride="carousel">

    <!--Carousel items-->
    <div class="relative w-full overflow-hidden after:clear-both after:block after:content-['']">
      <!--First item-->
      <div
        class="relative float-left -mr-[100%] w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-active data-twe-carousel-item style="backface-visibility: hidden">
        <div class="video-group">
          <video :ref="(el: any) => videos[0] = el" v-lazy src="@/assets/video/cross-id/0.mp4" muted loop
            autoplay></video>
          <div></div>
          <video :ref="(el: any) => videos[1] = el" v-lazy src="@/assets/video/cross-id/1.mp4" muted loop></video>
        </div>

      </div>
      <!--Second item-->
      <div
        class="relative float-left -mr-[100%] hidden w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-item style="backface-visibility: hidden">

        <div class="video-group">
          <video :ref="(el: any) => videos[2] = el" v-lazy src="@/assets/video/cross-id/1.mp4" muted loop></video>
          <div></div>
          <video :ref="(el: any) => videos[3] = el" v-lazy src="@/assets/video/cross-id/2.mp4" muted loop></video>
        </div>
      </div>
      <!--Third item-->
      <div
        class="relative float-left -mr-[100%] hidden w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-item style="backface-visibility: hidden">

        <div class="video-group">
          <video :ref="(el: any) => videos[4] = el" v-lazy src="@/assets/video/cross-id/2.mp4" muted loop></video>
          <div></div>
          <video :ref="(el: any) => videos[5] = el" v-lazy src="@/assets/video/cross-id/0.mp4" muted loop
            autoplay></video>
        </div>
      </div>
    </div>

    <!--Carousel indicators-->
    <div class="absolute bottom-0 left-0 right-0 z-[2] mx-[15%] -mb-8 flex list-none justify-center p-0"
      data-twe-carousel-indicators>
      <button type="button" data-twe-target="#crossIdCarousel" data-twe-slide-to="0" data-twe-carousel-active
        class="indicator" aria-current="true" aria-label="Slide 1"></button>
      <button type="button" data-twe-target="#crossIdCarousel" data-twe-slide-to="1" class="indicator"
        aria-label="Slide 2"></button>
      <button type="button" data-twe-target="#crossIdCarousel" data-twe-slide-to="2" class="indicator"
        aria-label="Slide 3"></button>
    </div>

    <!--Carousel controls - prev item-->
    <button class="indicator-btn indicator-left-btn" type="button" data-twe-target="#crossIdCarousel"
      data-twe-slide="prev">
      <span class="inline-block h-8 w-8">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
          class="h-6 w-6">
          <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
        </svg>
      </span>
      <span
        class="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Previous</span>
    </button>
    <!--Carousel controls - next item-->
    <button class="indicator-btn indicator-right-btn" type="button" data-twe-target="#crossIdCarousel"
      data-twe-slide="next">
      <span class="inline-block h-8 w-8">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
          class="h-6 w-6">
          <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
        </svg>
      </span>
      <span
        class="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Next</span>
    </button>
  </div>
</template>

<style scoped>
.video-group {
  video {
    width: 49%;
  }

  @media (max-width: 768px) {
    video {
      width: 100% !important;
    }

    div {
      width: 0;
    }
  }

  div {
    width: 1%;
  }

  * {
    @apply inline-block;
  }
}
</style>