// Config
// ------------
// Description: The configuration file for the website.

export interface Logo {
  src: string;
  alt: string;
}

export type Mode = "auto" | "light" | "dark";

export interface Config {
  siteTitle: string;
  siteDescription: string;
  ogImage: string;
  logo: Logo;
  canonical: boolean;
  noindex: boolean;
  mode: Mode;
  scrollAnimations: boolean;
}

export const configData: Config = {
  siteTitle: "Splinter Case Study",
  siteDescription:
    "Splinter is an open-source pipeline designed to simplify the processing of unstructured data and its integration into AI and machine learning applications",
  ogImage: "og.png",
  logo: {
    src: "favicon.svg",
    alt: "Splinter logo",
  },
  canonical: true,
  noindex: false,
  mode: "auto",
  scrollAnimations: true,
};
