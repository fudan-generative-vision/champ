<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import { initTWE, Carousel } from 'tw-elements';
import { inVisible } from '@/utils/video';
onMounted(() => {
  initTWE({ Carousel }, { allowReinits: true });

  comparisionsCarousel.value?.addEventListener('slide.twe.carousel', (v: any) => {
    const from = v.from;
    const to = v.to;
    comparisionVideos.value[from]?.pause();
    if (inVisible(comparisionVideos.value[from])) {
      comparisionVideos.value[to].play();
    }
  })
});

const comparisionsCarousel = ref<HTMLElement>();
const comparisionVideos = ref<HTMLVideoElement[]>([]);
// watch(comparisionsCarousel, (newV) => {
//   if (newV) {
//     newV.addEventListener('slide.twe.carousel', (v: any) => {
//       const from = v.from;
//       const to = v.to;
//       comparisionVideos.value[from]?.pause();
//       comparisionVideos.value[to]?.play();
//     })
//   }
// }, { once: true });
</script>

<template>
  <div ref="comparisionsCarousel" id="comparisionsCarousel" class="relative" data-twe-carousel-init
    data-twe-ride="carousel">

    <!--Carousel items-->
    <div class="relative w-full overflow-hidden after:clear-both after:block after:content-['']">
      <!--First item-->
      <div
        class="relative float-left -mr-[100%] w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-active data-twe-carousel-item style="backface-visibility: hidden">
        <video :ref="(el: any) => comparisionVideos[0] = el" v-lazy src="@/assets/video/comparisions/comparision-05.mp4"
          muted loop autoplay></video>

      </div>
      <!--Second item-->
      <div
        class="relative float-left -mr-[100%] hidden w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-item style="backface-visibility: hidden">

        <video :ref="(el: any) => comparisionVideos[1] = el" v-lazy src="@/assets/video/comparisions/comparision-07.mp4"
          muted loop></video>
      </div>
      <!--Third item-->
      <div
        class="relative float-left -mr-[100%] hidden w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-item style="backface-visibility: hidden">

        <video :ref="(el: any) => comparisionVideos[2] = el" v-lazy src="@/assets/video/comparisions/comparision-09.mp4"
          muted loop></video>
      </div>

      <!--Forth item-->
      <div
        class="relative float-left -mr-[100%] hidden w-full transition-transform duration-[600ms] ease-in-out motion-reduce:transition-none"
        data-twe-carousel-item style="backface-visibility: hidden">
        <video :ref="(el: any) => comparisionVideos[3] = el" v-lazy src="@/assets/video/comparisions/comparision-10.mp4"
          muted loop></video>
      </div>
    </div>

    <!--Carousel indicators-->
    <div class="absolute bottom-0 left-0 right-0 z-[2] mx-[15%] -mb-8 flex list-none justify-center p-0"
      data-twe-carousel-indicators>
      <button type="button" data-twe-target="#comparisionsCarousel" data-twe-slide-to="0" data-twe-carousel-active
        class="indicator" aria-current="true" aria-label="Slide 1"></button>
      <button type="button" data-twe-target="#comparisionsCarousel" data-twe-slide-to="1" class="indicator"
        aria-label="Slide 2"></button>
      <button type="button" data-twe-target="#comparisionsCarousel" data-twe-slide-to="2" class="indicator"
        aria-label="Slide 3"></button>
      <button type="button" data-twe-target="#comparisionsCarousel" data-twe-slide-to="3" class="indicator"
        aria-label="Slide 4"></button>
    </div>

    <!--Carousel controls - prev item-->
    <button class="indicator-btn indicator-left-btn" type="button" data-twe-target="#comparisionsCarousel"
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
    <button class="indicator-btn indicator-right-btn" type="button" data-twe-target="#comparisionsCarousel"
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